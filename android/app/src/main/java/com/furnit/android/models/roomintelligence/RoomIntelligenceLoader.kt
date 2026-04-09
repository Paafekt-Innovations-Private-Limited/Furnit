package com.furnit.android.models.roomintelligence

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import com.furnit.android.utils.LogUtil
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.max

object RoomIntelligenceLoader {
    private const val TAG = "RoomIntelligence"

    fun load(
        roomFile: File?,
        roomFolder: File?,
        roomWidthMeters: Float,
        roomHeightMeters: Float,
        roomDepthMeters: Float,
    ): RoomModel? {
        val fromMetadata = roomFile?.let { loadFromEnhancedMetadata(it) }
            ?: roomFile?.let { loadFromLegacyFlatMeta(it) }
        if (fromMetadata != null) return fromMetadata

        val folder = roomFolder ?: roomFile?.parentFile ?: return null
        return buildFallbackRoomModel(
            roomFolder = folder,
            roomWidthMeters = roomWidthMeters,
            roomHeightMeters = roomHeightMeters,
            roomDepthMeters = roomDepthMeters,
        )
    }

    private fun loadFromEnhancedMetadata(roomFile: File): RoomModel? {
        val stem = canonicalRoomStem(roomFile)
        val metadataFile = File(roomFile.parentFile, "$stem.room_metadata.json")
        if (!metadataFile.isFile) return null
        return try {
            val json = JSONObject(metadataFile.readText())
            parseEnhancedRoomMetadata(json)
        } catch (e: Exception) {
            LogUtil.w(TAG, "Failed to load enhanced metadata ${metadataFile.name}: ${e.message}")
            null
        }
    }

    private fun loadFromLegacyFlatMeta(roomFile: File): RoomModel? {
        val legacy = File(roomFile.parentFile, "${canonicalRoomStem(roomFile)}.${roomFile.extension}.meta")
        if (!legacy.isFile) return null
        return try {
            val json = JSONObject(legacy.readText())
            val values = mutableMapOf<String, String>()
            json.keys().forEach { key -> values[key] = json.optString(key) }
            buildLegacyRoomModel(values)
        } catch (e: Exception) {
            LogUtil.w(TAG, "Failed to load legacy room meta ${legacy.name}: ${e.message}")
            null
        }
    }

    private fun parseEnhancedRoomMetadata(json: JSONObject): RoomModel {
        val aabb = Aabb3(
            min = parseVec3(json.getJSONObject("aabbMin")),
            max = parseVec3(json.getJSONObject("aabbMax")),
        )
        val floor = parsePlane(json.getJSONObject("floor"))
        val ceiling = json.optJSONObject("ceiling")?.let { parsePlane(it) }
        val walls = json.optJSONArray("walls").mapArray { parsePlane(it as JSONObject) }
        val corners = json.optJSONArray("corners").mapArray {
            val obj = it as JSONObject
            RoomCorner(
                position = parseVec3(obj.getJSONObject("position")),
                uv = parseVec2(obj.getJSONObject("uv")),
            )
        }
        val freeRegions = json.optJSONArray("freeFloorRegions").mapArray {
            val obj = it as JSONObject
            val polygon = obj.optJSONArray("polygon").mapArray { parseVec2(it as JSONObject) }
            FreeFloorRegion(
                polygon = polygon,
                areaSqM = obj.optDouble("areaSqM", 0.0).toFloat(),
                uvBounds = FloorUvBounds(
                    min = parseVec2(obj.getJSONObject("uvBoundsMin")),
                    max = parseVec2(obj.getJSONObject("uvBoundsMax")),
                ),
                occupancyRatio = if (obj.has("occupancyRatio")) obj.optDouble("occupancyRatio", Double.NaN).toFloat().takeIf { it.isFinite() } else null,
            )
        }
        val palette = parseSurfacePalette(json.optJSONObject("surfacePalette"))
        val cameraInfo = json.optJSONObject("sourceCameraInfo")?.let { parseSourceCameraInfo(it) }
        val sceneToMeters = json.optDouble("sceneToMeters", 1.0).toFloat().takeIf { it.isFinite() && it > 0f } ?: 1f
        return RoomModel(
            aabb = aabb,
            floor = floor,
            ceiling = ceiling,
            walls = walls,
            corners = corners,
            freeFloorRegions = freeRegions,
            surfacePalette = palette,
            cameraInfo = cameraInfo,
            sceneToMeters = sceneToMeters,
        )
    }

    private fun buildLegacyRoomModel(legacy: Map<String, String>): RoomModel {
        val sceneWidth = legacy["roomSceneWidth"]?.toFloatOrNull() ?: 2f
        val sceneHeight = legacy["roomSceneHeight"]?.toFloatOrNull() ?: 2.5f
        val sceneDepth = legacy["roomSceneDepth"]?.toFloatOrNull() ?: 2f
        val roomHeightM = legacy["roomHeight"]?.toFloatOrNull()
        val sceneToMeters = if (roomHeightM != null && sceneHeight > 1e-4f) roomHeightM / sceneHeight else 1f
        val widthM = legacy["roomWidth"]?.toFloatOrNull() ?: sceneWidth * sceneToMeters
        val depthM = legacy["roomDepth"]?.toFloatOrNull() ?: sceneDepth * sceneToMeters
        val fallbackFolder = File(legacy["folderPath"].orEmpty())
        return buildFallbackRoomModel(
            roomFolder = fallbackFolder.takeIf { it.isDirectory },
            roomWidthMeters = widthM,
            roomHeightMeters = roomHeightM ?: sceneHeight * sceneToMeters,
            roomDepthMeters = depthM,
            sceneToMetersOverride = sceneToMeters,
            aabbOverride = Aabb3(
                min = Vec3f(0f, 0f, 0f),
                max = Vec3f(sceneWidth, sceneHeight, sceneDepth),
            ),
        )
    }

    fun buildFallbackRoomModel(
        roomFolder: File?,
        roomWidthMeters: Float,
        roomHeightMeters: Float,
        roomDepthMeters: Float,
        sceneToMetersOverride: Float? = null,
        aabbOverride: Aabb3? = null,
    ): RoomModel {
        val widthM = roomWidthMeters.takeIf { it.isFinite() && it > 0.05f } ?: 4f
        val heightM = roomHeightMeters.takeIf { it.isFinite() && it > 0.05f } ?: 3f
        val depthM = roomDepthMeters.takeIf { it.isFinite() && it > 0.05f } ?: 4.5f
        val sceneToMeters = sceneToMetersOverride?.takeIf { it.isFinite() && it > 0f } ?: 1f
        val aabb = aabbOverride ?: Aabb3(
            min = Vec3f(0f, 0f, 0f),
            max = Vec3f(widthM, heightM, depthM),
        )
        val floor = DetectedPlane(
            type = DetectedPlane.PlaneType.FLOOR,
            normal = Vec3f(0f, 1f, 0f),
            pointOnPlane = Vec3f(0f, 0f, 0f),
        )
        val ceiling = DetectedPlane(
            type = DetectedPlane.PlaneType.CEILING,
            normal = Vec3f(0f, -1f, 0f),
            pointOnPlane = Vec3f(widthM * 0.5f, heightM, depthM * 0.5f),
        )
        val walls = listOf(
            DetectedPlane(DetectedPlane.PlaneType.WALL, Vec3f(1f, 0f, 0f), Vec3f(0f, 0f, depthM * 0.5f)),
            DetectedPlane(DetectedPlane.PlaneType.WALL, Vec3f(-1f, 0f, 0f), Vec3f(widthM, 0f, depthM * 0.5f)),
            DetectedPlane(DetectedPlane.PlaneType.WALL, Vec3f(0f, 0f, 1f), Vec3f(widthM * 0.5f, 0f, 0f)),
            DetectedPlane(DetectedPlane.PlaneType.WALL, Vec3f(0f, 0f, -1f), Vec3f(widthM * 0.5f, 0f, depthM)),
        )
        val corners = listOf(
            RoomCorner(Vec3f(0f, 0f, 0f), Vec2f(0f, 0f)),
            RoomCorner(Vec3f(widthM, 0f, 0f), Vec2f(widthM, 0f)),
            RoomCorner(Vec3f(0f, 0f, depthM), Vec2f(0f, depthM)),
            RoomCorner(Vec3f(widthM, 0f, depthM), Vec2f(widthM, depthM)),
        )
        val freeRegion = FreeFloorRegion(
            polygon = listOf(
                Vec2f(0f, 0f),
                Vec2f(widthM, 0f),
                Vec2f(widthM, depthM),
                Vec2f(0f, depthM),
            ),
            areaSqM = widthM * depthM,
            uvBounds = FloorUvBounds(
                min = Vec2f(0f, 0f),
                max = Vec2f(widthM, depthM),
            ),
            occupancyRatio = 0f,
        )
        return RoomModel(
            aabb = aabb,
            floor = floor,
            ceiling = ceiling,
            walls = walls,
            corners = corners,
            freeFloorRegions = listOf(freeRegion),
            surfacePalette = sampleSurfacePalette(roomFolder),
            cameraInfo = null,
            sceneToMeters = sceneToMeters,
        )
    }

    private fun sampleSurfacePalette(roomFolder: File?): SurfacePalette {
        if (roomFolder == null || !roomFolder.isDirectory) return SurfacePalette.EMPTY
        return SurfacePalette(
            floor = sampleSurfaceColors(File(roomFolder, "floor.png")),
            walls = sampleSurfaceColors(File(roomFolder, "front_wall.png")),
            ceiling = sampleSurfaceColors(File(roomFolder, "ceiling.png")),
        )
    }

    private fun sampleSurfaceColors(file: File): SurfacePalette.SurfaceColors? {
        if (!file.isFile) return null
        val bitmap = decodeSampledBitmap(file, 160) ?: return null
        return bitmap.useSampledMeanColor()?.let { color ->
            SurfacePalette.SurfaceColors(primary = color, secondary = null, hint = SurfacePalette.MaterialHint.UNKNOWN)
        }
    }

    private fun decodeSampledBitmap(file: File, targetMaxSizePx: Int): Bitmap? {
        val bounds = BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

        var inSampleSize = 1
        var width = bounds.outWidth
        var height = bounds.outHeight
        while (max(width, height) > targetMaxSizePx) {
            width /= 2
            height /= 2
            inSampleSize *= 2
        }

        val options = BitmapFactory.Options().apply {
            this.inSampleSize = inSampleSize
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        return BitmapFactory.decodeFile(file.absolutePath, options)
    }

    private fun Bitmap.useSampledMeanColor(): Vec3f? {
        val width = width
        val height = height
        if (width <= 0 || height <= 0) return null
        val sampleStride = max(1, max(width, height) / 96)
        val pixels = IntArray(width * height)
        getPixels(pixels, 0, width, 0, 0, width, height)
        var sumR = 0.0
        var sumG = 0.0
        var sumB = 0.0
        var count = 0
        var y = 0
        while (y < height) {
            var x = 0
            while (x < width) {
                val pixel = pixels[y * width + x]
                sumR += ((pixel shr 16) and 0xFF) / 255.0
                sumG += ((pixel shr 8) and 0xFF) / 255.0
                sumB += (pixel and 0xFF) / 255.0
                count++
                x += sampleStride
            }
            y += sampleStride
        }
        if (count <= 0) return null
        return Vec3f((sumR / count).toFloat(), (sumG / count).toFloat(), (sumB / count).toFloat())
    }

    private fun parseVec3(json: JSONObject): Vec3f {
        return Vec3f(
            x = json.optDouble("x", 0.0).toFloat(),
            y = json.optDouble("y", 0.0).toFloat(),
            z = json.optDouble("z", 0.0).toFloat(),
        )
    }

    private fun parseVec2(json: JSONObject): Vec2f {
        return Vec2f(
            x = json.optDouble("x", 0.0).toFloat(),
            y = json.optDouble("y", 0.0).toFloat(),
        )
    }

    private fun parsePlane(json: JSONObject): DetectedPlane {
        val type = when (json.optString("type", "wall")) {
            "floor" -> DetectedPlane.PlaneType.FLOOR
            "ceiling" -> DetectedPlane.PlaneType.CEILING
            else -> DetectedPlane.PlaneType.WALL
        }
        return DetectedPlane(
            type = type,
            normal = parseVec3(json.getJSONObject("normal")),
            pointOnPlane = parseVec3(json.getJSONObject("pointOnPlane")),
        )
    }

    private fun parseSurfacePalette(json: JSONObject?): SurfacePalette {
        if (json == null) return SurfacePalette.EMPTY
        return SurfacePalette(
            floor = json.optJSONObject("floor")?.let(::parseSurfaceEntry),
            walls = json.optJSONObject("walls")?.let(::parseSurfaceEntry),
            ceiling = json.optJSONObject("ceiling")?.let(::parseSurfaceEntry),
        )
    }

    private fun parseSurfaceEntry(json: JSONObject): SurfacePalette.SurfaceColors {
        val hint = when (json.optString("hint", "unknown")) {
            "wood" -> SurfacePalette.MaterialHint.WOOD
            "tile" -> SurfacePalette.MaterialHint.TILE
            "carpet" -> SurfacePalette.MaterialHint.CARPET
            "concrete" -> SurfacePalette.MaterialHint.CONCRETE
            "brick" -> SurfacePalette.MaterialHint.BRICK
            "plaster" -> SurfacePalette.MaterialHint.PLASTER
            "marble" -> SurfacePalette.MaterialHint.MARBLE
            else -> SurfacePalette.MaterialHint.UNKNOWN
        }
        return SurfacePalette.SurfaceColors(
            primary = parseVec3(json.getJSONObject("primary")),
            secondary = json.optJSONObject("secondary")?.let(::parseVec3),
            hint = hint,
        )
    }

    private fun parseSourceCameraInfo(json: JSONObject): SourceCameraInfo {
        fun optFloat(key: String): Float? =
            if (!json.has(key)) null else json.optDouble(key, Double.NaN).toFloat().takeIf { it.isFinite() }
        fun optInt(key: String): Int? =
            if (!json.has(key)) null else json.optInt(key)
        return SourceCameraInfo(
            focalLengthMm = optFloat("focalLengthMM"),
            sensorWidthMm = optFloat("sensorWidthMM"),
            focalLength35mmEquivalentMm = optFloat("focalLength35mmEquivalentMM"),
            subjectDistanceMeters = optFloat("subjectDistanceMeters"),
            imageWidthPx = optInt("imageWidthPx"),
            imageHeightPx = optInt("imageHeightPx"),
            photoOrientation = optInt("photoOrientation"),
        )
    }

    private fun canonicalRoomStem(roomFile: File): String {
        val stem = roomFile.nameWithoutExtension
        return when {
            stem.endsWith("_classic") -> stem.removeSuffix("_classic")
            else -> stem
        }
    }

    private fun <T> JSONArray?.mapArray(transform: (Any) -> T): List<T> {
        if (this == null) return emptyList()
        val out = ArrayList<T>(length())
        for (idx in 0 until length()) {
            out += transform(get(idx))
        }
        return out
    }
}
