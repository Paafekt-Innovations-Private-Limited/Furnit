package com.furnit.android.models

import android.content.Context
import android.util.Log
import java.io.File

class ModelManager(private val context: Context) {
    private val models = mutableListOf<Model>()

    companion object {
        private const val TAG = "ModelManager"
        private const val ROOMS_DIR = "rooms"
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
        val roomsDir = File(context.filesDir, ROOMS_DIR)
        if (!roomsDir.exists()) {
            Log.d(TAG, "No rooms directory found")
            return
        }

        val roomFolders = roomsDir.listFiles { file -> file.isDirectory } ?: return
        Log.d(TAG, "Found ${roomFolders.size} user-created rooms")

        val userRooms = mutableListOf<Model>()

        for (folder in roomFolders) {
            val frontWall = File(folder, "front_wall.png")
            val glbFile = File(folder, "room.glb")
            val metadataFile = File(folder, "metadata.txt")

            if (frontWall.exists() || glbFile.exists()) {
                // Read room name and created timestamp from metadata
                var roomName = "My Room ${folder.name.substringAfter("room_")}"
                var createdAt = folder.lastModified() // fallback to folder modification time

                if (metadataFile.exists()) {
                    try {
                        val lines = metadataFile.readLines()
                        lines.firstOrNull { it.startsWith("name=") }
                            ?.substringAfter("name=")?.let { roomName = it }
                        lines.firstOrNull { it.startsWith("created=") }
                            ?.substringAfter("created=")?.toLongOrNull()?.let { createdAt = it }
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to read metadata for ${folder.name}", e)
                    }
                }

                // Use GLB file path if it exists, otherwise use folder path
                val assetPath = if (glbFile.exists()) {
                    glbFile.absolutePath
                } else {
                    folder.absolutePath
                }

                val model = Model(
                    id = folder.name,
                    name = roomName,
                    assetPath = assetPath,
                    isUserCreated = true,
                    thumbnailPath = if (frontWall.exists()) frontWall.absolutePath else null,
                    createdAt = createdAt
                )
                userRooms.add(model)
                Log.d(TAG, "Loaded user room: ${model.name} at ${model.assetPath} (GLB: ${glbFile.exists()}, created: $createdAt)")
            }
        }

        // Sort user rooms by creation date descending (newest first)
        userRooms.sortByDescending { it.createdAt }
        models.addAll(userRooms)
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
