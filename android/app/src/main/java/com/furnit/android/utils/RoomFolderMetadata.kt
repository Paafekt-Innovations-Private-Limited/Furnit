package com.furnit.android.utils

import com.furnit.android.utils.LogUtil
import org.json.JSONObject
import java.io.File

/**
 * Single source of truth for per-room viewer + list fields on disk.
 *
 * - **room_meta.json** — canonical (written on Sharp generation, save, and lazily migrated from metadata.txt).
 * - **metadata.txt** — legacy key=value; still written for older tooling; read when JSON is missing.
 *
 * [readFromFolder] is used by [com.furnit.android.models.ModelManager] (list) and
 * [com.furnit.android.SharpRoomActivity] (viewer) so orientation/dims always match.
 */
object RoomFolderMetadata {

    private const val TAG = "RoomFolderMetadata"
    const val JSON_FILE_NAME = "room_meta.json"
    private const val LEGACY_TXT = "metadata.txt"
    private const val SCHEMA_VERSION = 1

    data class Snapshot(
        val name: String? = null,
        val createdAt: Long? = null,
        val type: String? = null,
        val photoOrientation: String = "portrait",
        val photoWideAngle: Boolean = false,
        val roomWidth: Float? = null,
        val roomHeight: Float? = null,
        val roomDepth: Float? = null,
        val roomCenterX: Float? = null,
        val roomCenterY: Float? = null,
        val roomCenterZ: Float? = null,
    ) {
        fun normalizedOrientation(): String =
            if (photoOrientation.trim().lowercase() == "landscape") "landscape" else "portrait"
    }

    /**
     * Read JSON if present; else parse legacy [metadata.txt]. If only txt existed, write JSON once (migration).
     */
    fun readFromFolder(folder: File): Snapshot? {
        if (!folder.isDirectory) return null
        val jsonFile = File(folder, JSON_FILE_NAME)
        if (jsonFile.exists()) {
            try {
                parseJson(jsonFile.readText())?.let { return it }
            } catch (e: Exception) {
                LogUtil.w(TAG, "Failed to parse $JSON_FILE_NAME in ${folder.name}", e)
            }
        }
        val fromTxt = parseMetadataTxt(File(folder, LEGACY_TXT)) ?: return null
        if (!jsonFile.exists()) {
            try {
                writeToFolder(folder, fromTxt)
                LogUtil.d(TAG, "Migrated $LEGACY_TXT → $JSON_FILE_NAME in ${folder.name}")
            } catch (e: Exception) {
                LogUtil.w(TAG, "Could not write $JSON_FILE_NAME after txt parse", e)
            }
        }
        return fromTxt
    }

    fun writeToFolder(folder: File, snapshot: Snapshot) {
        if (!folder.exists()) folder.mkdirs()
        val jo = JSONObject()
        jo.put("schemaVersion", SCHEMA_VERSION)
        snapshot.name?.takeIf { it.isNotBlank() }?.let { jo.put("name", it) }
        snapshot.createdAt?.let { jo.put("created", it) }
        snapshot.type?.takeIf { it.isNotBlank() }?.let { jo.put("type", it) }
        jo.put("photoOrientation", snapshot.normalizedOrientation())
        jo.put("photoWideAngle", snapshot.photoWideAngle)
        snapshot.roomWidth?.let { if (it > 0f) jo.put("roomWidth", it.toDouble()) }
        snapshot.roomHeight?.let { if (it > 0f) jo.put("roomHeight", it.toDouble()) }
        snapshot.roomDepth?.let { if (it > 0f) jo.put("roomDepth", it.toDouble()) }
        snapshot.roomCenterX?.let { jo.put("roomCenterX", it.toDouble()) }
        snapshot.roomCenterY?.let { jo.put("roomCenterY", it.toDouble()) }
        snapshot.roomCenterZ?.let { jo.put("roomCenterZ", it.toDouble()) }
        File(folder, JSON_FILE_NAME).writeText(jo.toString())
    }

    private fun parseJson(text: String): Snapshot? {
        val jo = JSONObject(text)
        fun optFloat(key: String): Float? {
            if (!jo.has(key)) return null
            val d = jo.optDouble(key, Double.NaN)
            return if (d.isNaN()) null else d.toFloat()
        }
        val rawOrient = jo.optString("photoOrientation", "portrait").trim().lowercase()
        return Snapshot(
            name = jo.optString("name", "").takeIf { it.isNotBlank() },
            createdAt = if (jo.has("created")) jo.getLong("created") else null,
            type = jo.optString("type", "").takeIf { it.isNotBlank() },
            photoOrientation = if (rawOrient == "landscape") "landscape" else "portrait",
            photoWideAngle = jo.optBoolean("photoWideAngle", false),
            roomWidth = optFloat("roomWidth"),
            roomHeight = optFloat("roomHeight"),
            roomDepth = optFloat("roomDepth"),
            roomCenterX = optFloat("roomCenterX"),
            roomCenterY = optFloat("roomCenterY"),
            roomCenterZ = optFloat("roomCenterZ"),
        )
    }

    private fun parseMetadataTxt(file: File): Snapshot? {
        if (!file.exists()) return null
        val map = linkedMapOf<String, String>()
        try {
            file.readLines().forEach { line ->
                val idx = line.indexOf('=')
                if (idx > 0) {
                    val key = line.substring(0, idx).trim()
                    map[key] = line.substring(idx + 1).trim()
                }
            }
        } catch (e: Exception) {
            LogUtil.w(TAG, "Failed to read ${file.name}", e)
            return null
        }
        if (map.isEmpty()) return null
        val rawOrient = map["photoOrientation"]?.trim()?.lowercase() ?: "portrait"
        val orient = if (rawOrient == "landscape") "landscape" else "portrait"
        val wideRaw = map["photoWideAngle"]?.trim()?.lowercase()
        return Snapshot(
            name = map["name"],
            createdAt = map["created"]?.toLongOrNull(),
            type = map["type"],
            photoOrientation = orient,
            photoWideAngle = wideRaw == "true",
            roomWidth = map["roomWidth"]?.toFloatOrNull(),
            roomHeight = map["roomHeight"]?.toFloatOrNull(),
            roomDepth = map["roomDepth"]?.toFloatOrNull(),
            roomCenterX = map["roomCenterX"]?.toFloatOrNull(),
            roomCenterY = map["roomCenterY"]?.toFloatOrNull(),
            roomCenterZ = map["roomCenterZ"]?.toFloatOrNull(),
        )
    }
}
