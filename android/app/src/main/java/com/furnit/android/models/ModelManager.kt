package com.furnit.android.models

import android.content.Context
import com.furnit.android.utils.LogUtil
import com.furnit.android.utils.RoomFolderMetadata
import java.io.File

class ModelManager(private val context: Context) {
    private val models = mutableListOf<Model>()

    companion object {
        private const val TAG = "ModelManager"
        private const val ROOMS_DIR = "rooms"
        /** Built-in demo GLBs under `assets/bundled_rooms/`. */
        const val BUNDLED_ROOM_ASSETS_DIR = "bundled_rooms"

        val BUNDLED_ROOM_IDS = setOf("scandinavian_minimal", "industrial_loft")

        fun isBundledRoomId(roomId: String?): Boolean =
            roomId != null && roomId in BUNDLED_ROOM_IDS

        private fun normalizeRoomName(roomName: String): String {
            return roomName.trim()
                .replace("\\s+".toRegex(), " ")
                .lowercase()
        }

        fun isRoomNameAvailable(context: Context, candidateName: String, excludeRoomId: String? = null): Boolean {
            return ModelManager(context).isRoomNameAvailable(candidateName, excludeRoomId)
        }

        fun findAvailableRoomName(context: Context, preferredName: String, excludeRoomId: String? = null): String {
            return ModelManager(context).findAvailableRoomName(preferredName, excludeRoomId)
        }
    }

    init {
        loadModels()
    }

    private fun loadModels() {
        models.clear()

        // Load user-created rooms first (sorted by newest first)
        loadUserCreatedRooms()

        // Add bundled models at the bottom
        models.add(
            Model(
                id = "scandinavian_minimal",
                name = "Scandinavian Minimal",
                assetPath = "$BUNDLED_ROOM_ASSETS_DIR/scandinavian_minimal.glb",
                createdAt = 0L,
                photoOrientation = "landscape",
            ),
        )
        models.add(
            Model(
                id = "industrial_loft",
                name = "Industrial Loft",
                assetPath = "$BUNDLED_ROOM_ASSETS_DIR/industrial_loft.glb",
                createdAt = 0L,
                photoOrientation = "landscape",
            ),
        )
    }

    private fun loadUserCreatedRooms() {
        val userRooms = mutableListOf<Model>()

        // Load from regular rooms directory
        loadRoomsFromDir(File(context.filesDir, ROOMS_DIR), userRooms)

        // Sort user rooms by creation date descending (newest first)
        userRooms.sortByDescending { it.createdAt }
        models.addAll(userRooms)
    }

    private fun loadRoomsFromDir(roomsDir: File, userRooms: MutableList<Model>) {
        if (!roomsDir.exists()) {
            LogUtil.d(TAG, "Directory not found: ${roomsDir.name}")
            return
        }

        val roomFolders = roomsDir.listFiles { file -> file.isDirectory } ?: return
        LogUtil.d(TAG, "Found ${roomFolders.size} rooms in ${roomsDir.name}")

        for (folder in roomFolders) {
            val frontWall = File(folder, "front_wall.png")
            val thumbnail = File(folder, "thumbnail.png")
            val glbFile = File(folder, "room.glb")

            if (frontWall.exists() || glbFile.exists()) {
                // Read room name and created timestamp from metadata
                var roomName = "My Room ${folder.name.substringAfter("room_")}"
                var createdAt = folder.lastModified() // fallback to folder modification time

                // Room dimension variables
                var roomWidth: Float? = null
                var roomHeight: Float? = null
                var roomDepth: Float? = null
                var roomCenterX: Float? = null
                var roomCenterY: Float? = null
                var roomCenterZ: Float? = null
                var photoOrientation = "portrait"
                var photoWideAngle = false

                val disk = RoomFolderMetadata.readFromFolder(folder)
                if (disk != null && disk.previewOnly == true) {
                    LogUtil.d(TAG, "Skipping preview-only room (not saved to library): ${folder.name}")
                    continue
                }
                if (disk != null) {
                    disk.name?.takeIf { it.isNotBlank() }?.let { roomName = it }
                    disk.createdAt?.let { createdAt = it }
                    val arSc = disk.arDisplayScale?.takeIf { it > 0f } ?: 1f
                    roomWidth = disk.roomWidth?.let { it * arSc }
                    roomHeight = disk.roomHeight?.let { it * arSc }
                    roomDepth = disk.roomDepth?.let { it * arSc }
                    roomCenterX = disk.roomCenterX?.let { it * arSc }
                    roomCenterY = disk.roomCenterY?.let { it * arSc }
                    roomCenterZ = disk.roomCenterZ?.let { it * arSc }
                    photoOrientation = disk.normalizedOrientation()
                    photoWideAngle = disk.photoWideAngle
                }

                // Use GLB file path if it exists, otherwise use folder path.
                val assetPath = when {
                    glbFile.exists() -> glbFile.absolutePath
                    else -> folder.absolutePath
                }

                // Use thumbnail.png (AI rooms) or front_wall.png (regular rooms)
                val thumbPath = when {
                    thumbnail.exists() -> thumbnail.absolutePath
                    frontWall.exists() -> frontWall.absolutePath
                    else -> null
                }

                val model = Model(
                    id = folder.name,
                    name = roomName,
                    assetPath = assetPath,
                    isUserCreated = true,
                    thumbnailPath = thumbPath,
                    createdAt = createdAt,
                    roomWidth = roomWidth,
                    roomHeight = roomHeight,
                    roomDepth = roomDepth,
                    roomCenterX = roomCenterX,
                    roomCenterY = roomCenterY,
                    roomCenterZ = roomCenterZ,
                    photoOrientation = photoOrientation,
                    photoWideAngle = photoWideAngle
                )
                userRooms.add(model)
                LogUtil.d(TAG, "Loaded room: ${model.name} at ${model.assetPath} (created: $createdAt, dims: ${roomWidth}x${roomHeight}x${roomDepth}, photoOrientation: $photoOrientation)")
            }
        }
    }

    fun listModels(): List<Model> = models.toList()

    fun getModel(id: String): Model? = models.find { it.id == id }

    fun refresh() {
        loadModels()
    }

    fun isRoomNameAvailable(candidateName: String, excludeRoomId: String? = null): Boolean {
        val normalizedCandidateName = normalizeRoomName(candidateName)
        if (normalizedCandidateName.isEmpty()) return false
        return models.none { model ->
            model.id != excludeRoomId && normalizeRoomName(model.name) == normalizedCandidateName
        }
    }

    fun findAvailableRoomName(preferredName: String, excludeRoomId: String? = null): String {
        val baseRoomName = preferredName.trim().replace("\\s+".toRegex(), " ").ifEmpty { "Room" }
        if (isRoomNameAvailable(baseRoomName, excludeRoomId)) return baseRoomName

        var suffixIndex = 2
        while (true) {
            val candidateName = "$baseRoomName ($suffixIndex)"
            if (isRoomNameAvailable(candidateName, excludeRoomId)) {
                return candidateName
            }
            suffixIndex++
        }
    }

    fun deleteRoom(id: String): Boolean {
        if (isBundledRoomId(id)) return false
        val model = models.find { it.id == id && it.isUserCreated } ?: return false
        // assetPath may be a GLB file or a folder - get the parent folder if it's a file
        val folder = File(model.assetPath).let {
            if (it.isFile) it.parentFile else it
        }
        return try {
            folder?.deleteRecursively()
            models.remove(model)
            LogUtil.d(TAG, "Deleted room: $id")
            true
        } catch (e: Exception) {
            LogUtil.e(TAG, "Failed to delete room: $id", e)
            false
        }
    }

    /**
     * Updates the display name in [RoomFolderMetadata] for a user room folder.
     */
    fun renameUserRoom(roomId: String, newName: String): Boolean {
        if (isBundledRoomId(roomId)) return false
        val trimmed = newName.trim()
        if (trimmed.isEmpty()) return false
        if (!isRoomNameAvailable(trimmed, excludeRoomId = roomId)) return false
        val model = models.find { it.id == roomId && it.isUserCreated } ?: return false
        val folder = File(model.assetPath).let { if (it.isFile) it.parentFile else it } ?: return false
        return try {
            val base = RoomFolderMetadata.readFromFolder(folder)
                ?: RoomFolderMetadata.Snapshot(
                    name = trimmed,
                    createdAt = folder.lastModified(),
                )
            val next = RoomFolderMetadata.snapshotPreservingCalibrationFields(folder, base.copy(name = trimmed))
            RoomFolderMetadata.writeToFolder(folder, next)
            loadModels()
            LogUtil.d(TAG, "Renamed room $roomId to \"$trimmed\"")
            true
        } catch (e: Exception) {
            LogUtil.e(TAG, "Failed to rename room: $roomId", e)
            false
        }
    }
}
