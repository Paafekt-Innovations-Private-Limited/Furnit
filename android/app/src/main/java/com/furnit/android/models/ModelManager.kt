package com.furnit.android.models

import android.content.Context
import android.util.Log
import java.io.File

class ModelManager(private val context: Context) {
    private val models = mutableListOf<Model>()

    companion object {
        private const val TAG = "ModelManager"
        private const val ROOMS_DIR = "rooms"
        private const val SHARP_ROOMS_DIR = "sharp_rooms"
    }

    init {
        loadModels()
    }

    private fun loadModels() {
        models.clear()

        // Load user-created rooms first (sorted by newest first)
        loadUserCreatedRooms()

        // Add bundled models at the bottom
        models.add(Model("vintage", "Vintage Living Room", "models/vintage.glb", createdAt = 0L))
        models.add(Model("cozy_room", "Cozy Living Room", "models/cozy_room.glb", createdAt = 0L))
    }

    private fun loadUserCreatedRooms() {
        val userRooms = mutableListOf<Model>()

        // Load from regular rooms directory
        loadRoomsFromDir(File(context.filesDir, ROOMS_DIR), userRooms)

        // Load from sharp_rooms directory (AI-generated rooms)
        loadRoomsFromDir(File(context.filesDir, SHARP_ROOMS_DIR), userRooms)

        // Sort user rooms by creation date descending (newest first)
        userRooms.sortByDescending { it.createdAt }
        models.addAll(userRooms)
    }

    private fun loadRoomsFromDir(roomsDir: File, userRooms: MutableList<Model>) {
        if (!roomsDir.exists()) {
            Log.d(TAG, "Directory not found: ${roomsDir.name}")
            return
        }

        val roomFolders = roomsDir.listFiles { file -> file.isDirectory } ?: return
        Log.d(TAG, "Found ${roomFolders.size} rooms in ${roomsDir.name}")

        for (folder in roomFolders) {
            val frontWall = File(folder, "front_wall.png")
            val thumbnail = File(folder, "thumbnail.png")
            val glbFile = File(folder, "room.glb")
            val plyFile = File(folder, "room.ply")
            val metadataFile = File(folder, "metadata.txt")

            if (frontWall.exists() || glbFile.exists() || plyFile.exists()) {
                // Read room name and created timestamp from metadata
                var roomName = "My Room ${folder.name.substringAfter("room_")}"
                var createdAt = folder.lastModified() // fallback to folder modification time

                // Room dimension variables
                var roomWidth: Float? = null
                var roomHeight: Float? = null
                var roomDepth: Float? = null
                var photoOrientation = "portrait"

                if (metadataFile.exists()) {
                    try {
                        val lines = metadataFile.readLines()
                        lines.firstOrNull { it.startsWith("name=") }
                            ?.substringAfter("name=")?.let { roomName = it }
                        lines.firstOrNull { it.startsWith("created=") }
                            ?.substringAfter("created=")?.toLongOrNull()?.let { createdAt = it }
                        // Load room dimensions from metadata
                        lines.firstOrNull { it.startsWith("roomWidth=") }
                            ?.substringAfter("roomWidth=")?.toFloatOrNull()?.let { roomWidth = it }
                        lines.firstOrNull { it.startsWith("roomHeight=") }
                            ?.substringAfter("roomHeight=")?.toFloatOrNull()?.let { roomHeight = it }
                        lines.firstOrNull { it.startsWith("roomDepth=") }
                            ?.substringAfter("roomDepth=")?.toFloatOrNull()?.let { roomDepth = it }
                        lines.firstOrNull { it.startsWith("photoOrientation=") }
                            ?.substringAfter("photoOrientation=")?.let { photoOrientation = it }
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to read metadata for ${folder.name}", e)
                    }
                }

                // Use GLB/PLY file path if it exists, otherwise use folder path
                val assetPath = when {
                    glbFile.exists() -> glbFile.absolutePath
                    plyFile.exists() -> plyFile.absolutePath
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
                    photoOrientation = photoOrientation
                )
                userRooms.add(model)
                Log.d(TAG, "Loaded room: ${model.name} at ${model.assetPath} (created: $createdAt, dims: ${roomWidth}x${roomHeight}x${roomDepth})")
            }
        }
    }

    fun listModels(): List<Model> = models.toList()

    fun getModel(id: String): Model? = models.find { it.id == id }

    fun refresh() {
        loadModels()
    }

    fun deleteRoom(id: String): Boolean {
        val model = models.find { it.id == id && it.isUserCreated } ?: return false
        // assetPath may be a GLB file or a folder - get the parent folder if it's a file
        val folder = File(model.assetPath).let {
            if (it.isFile) it.parentFile else it
        }
        return try {
            folder?.deleteRecursively()
            models.remove(model)
            Log.d(TAG, "Deleted room: $id")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to delete room: $id", e)
            false
        }
    }
}
