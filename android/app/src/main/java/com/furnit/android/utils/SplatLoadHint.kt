package com.furnit.android.utils

import org.json.JSONObject
import java.io.File

data class SplatLoadHintVector3(
    val x: Float,
    val y: Float,
    val z: Float,
) {
    fun toJson(): JSONObject =
        JSONObject()
            .put("x", x.toDouble())
            .put("y", y.toDouble())
            .put("z", z.toDouble())

    companion object {
        fun fromJson(jsonObject: JSONObject?): SplatLoadHintVector3? {
            if (jsonObject == null) return null
            return SplatLoadHintVector3(
                x = jsonObject.optDouble("x", Double.NaN).toFloat(),
                y = jsonObject.optDouble("y", Double.NaN).toFloat(),
                z = jsonObject.optDouble("z", Double.NaN).toFloat(),
            ).takeIf { it.x.isFinite() && it.y.isFinite() && it.z.isFinite() }
        }
    }
}

data class SplatLoadHint(
    val schemaVersion: Int = CURRENT_SCHEMA_VERSION,
    val fileByteCount: Long,
    val fileModificationTimeEpochMillis: Long,
    val splatCount: Int,
    val fullBoundsMin: SplatLoadHintVector3,
    val fullBoundsMax: SplatLoadHintVector3,
    val framingBoundsMin: SplatLoadHintVector3,
    val framingBoundsMax: SplatLoadHintVector3,
    val centroid: SplatLoadHintVector3,
) {
    fun matches(file: File): Boolean {
        if (!file.exists()) return false
        return file.length() == fileByteCount && file.lastModified() == fileModificationTimeEpochMillis
    }

    fun toJson(): JSONObject =
        JSONObject()
            .put("schemaVersion", schemaVersion)
            .put("fileByteCount", fileByteCount)
            .put("fileModificationTimeEpochMillis", fileModificationTimeEpochMillis)
            .put("splatCount", splatCount)
            .put("fullBoundsMin", fullBoundsMin.toJson())
            .put("fullBoundsMax", fullBoundsMax.toJson())
            .put("framingBoundsMin", framingBoundsMin.toJson())
            .put("framingBoundsMax", framingBoundsMax.toJson())
            .put("centroid", centroid.toJson())

    companion object {
        const val CURRENT_SCHEMA_VERSION = 1

        fun sidecarFileFor(roomPlyFile: File): File {
            val roomFolder = roomPlyFile.parentFile ?: return File(roomPlyFile.absolutePath + ".splat_load_hint.json")
            val stem = roomPlyFile.name.substringBeforeLast('.')
            return File(roomFolder, "$stem.splat_load_hint.json")
        }

        fun createForFile(
            roomPlyFile: File,
            splatCount: Int,
            fullBoundsMin: SplatLoadHintVector3,
            fullBoundsMax: SplatLoadHintVector3,
            framingBoundsMin: SplatLoadHintVector3 = fullBoundsMin,
            framingBoundsMax: SplatLoadHintVector3 = fullBoundsMax,
            centroid: SplatLoadHintVector3,
        ): SplatLoadHint? {
            if (!roomPlyFile.exists()) return null
            return SplatLoadHint(
                fileByteCount = roomPlyFile.length(),
                fileModificationTimeEpochMillis = roomPlyFile.lastModified(),
                splatCount = splatCount,
                fullBoundsMin = fullBoundsMin,
                fullBoundsMax = fullBoundsMax,
                framingBoundsMin = framingBoundsMin,
                framingBoundsMax = framingBoundsMax,
                centroid = centroid,
            )
        }

        fun readFrom(sidecarFile: File): SplatLoadHint? {
            if (!sidecarFile.exists()) return null
            return try {
                fromJson(JSONObject(sidecarFile.readText()))
            } catch (exception: Exception) {
                LogUtil.w("SplatLoadHint", "Failed to read ${sidecarFile.absolutePath}", exception)
                null
            }
        }

        fun writeTo(sidecarFile: File, hint: SplatLoadHint) {
            val parentDirectory = sidecarFile.parentFile
            if (parentDirectory != null && !parentDirectory.exists()) {
                parentDirectory.mkdirs()
            }
            sidecarFile.writeText(hint.toJson().toString())
        }

        fun fromJson(jsonObject: JSONObject): SplatLoadHint? {
            if (jsonObject.optInt("schemaVersion", CURRENT_SCHEMA_VERSION) != CURRENT_SCHEMA_VERSION) return null
            val fullBoundsMin = SplatLoadHintVector3.fromJson(jsonObject.optJSONObject("fullBoundsMin")) ?: return null
            val fullBoundsMax = SplatLoadHintVector3.fromJson(jsonObject.optJSONObject("fullBoundsMax")) ?: return null
            val framingBoundsMin = SplatLoadHintVector3.fromJson(jsonObject.optJSONObject("framingBoundsMin")) ?: fullBoundsMin
            val framingBoundsMax = SplatLoadHintVector3.fromJson(jsonObject.optJSONObject("framingBoundsMax")) ?: fullBoundsMax
            val centroid = SplatLoadHintVector3.fromJson(jsonObject.optJSONObject("centroid")) ?: return null
            return SplatLoadHint(
                schemaVersion = jsonObject.optInt("schemaVersion", CURRENT_SCHEMA_VERSION),
                fileByteCount = jsonObject.optLong("fileByteCount", -1L),
                fileModificationTimeEpochMillis = jsonObject.optLong("fileModificationTimeEpochMillis", -1L),
                splatCount = jsonObject.optInt("splatCount", 0),
                fullBoundsMin = fullBoundsMin,
                fullBoundsMax = fullBoundsMax,
                framingBoundsMin = framingBoundsMin,
                framingBoundsMax = framingBoundsMax,
                centroid = centroid,
            ).takeIf {
                it.fileByteCount >= 0L &&
                    it.fileModificationTimeEpochMillis >= 0L &&
                    it.splatCount >= 0
            }
        }
    }
}
