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

        // Load bundled models from assets
        models.add(Model("vintage", "Vintage Living Room", "models/vintage.glb"))
        models.add(Model("cozy_room", "Cozy Living Room", "models/cozy_room.glb"))

        // Load user-created rooms from file system
        loadUserCreatedRooms()
    }

    private fun loadUserCreatedRooms() {
        val roomsDir = File(context.filesDir, ROOMS_DIR)
        if (!roomsDir.exists()) {
            Log.d(TAG, "No rooms directory found")
            return
        }

        val roomFolders = roomsDir.listFiles { file -> file.isDirectory } ?: return
        Log.d(TAG, "Found ${roomFolders.size} user-created rooms")

        for (folder in roomFolders) {
            val frontWall = File(folder, "front_wall.png")
            val glbFile = File(folder, "room.glb")
            val metadataFile = File(folder, "metadata.txt")

            if (frontWall.exists() || glbFile.exists()) {
                // Read room name from metadata or use folder name
                val roomName = if (metadataFile.exists()) {
                    try {
                        metadataFile.readLines().firstOrNull { it.startsWith("name=") }
                            ?.substringAfter("name=") ?: "My Room"
                    } catch (e: Exception) {
                        "My Room"
                    }
                } else {
                    "My Room ${folder.name.substringAfter("room_")}"
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
                    thumbnailPath = if (frontWall.exists()) frontWall.absolutePath else null
                )
                models.add(model)
                Log.d(TAG, "Loaded user room: ${model.name} at ${model.assetPath} (GLB: ${glbFile.exists()})")
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
