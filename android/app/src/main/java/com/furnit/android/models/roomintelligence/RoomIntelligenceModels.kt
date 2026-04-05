package com.furnit.android.models.roomintelligence

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class Vec2f(
    val x: Float,
    val y: Float,
) {
    operator fun plus(other: Vec2f) = Vec2f(x + other.x, y + other.y)
    operator fun minus(other: Vec2f) = Vec2f(x - other.x, y - other.y)
    operator fun times(scale: Float) = Vec2f(x * scale, y * scale)
}

data class Vec3f(
    val x: Float,
    val y: Float,
    val z: Float,
) {
    operator fun plus(other: Vec3f) = Vec3f(x + other.x, y + other.y, z + other.z)
    operator fun minus(other: Vec3f) = Vec3f(x - other.x, y - other.y, z - other.z)
    operator fun times(scale: Float) = Vec3f(x * scale, y * scale, z * scale)

    fun dot(other: Vec3f): Float = x * other.x + y * other.y + z * other.z

    fun lengthSquared(): Float = dot(this)

    fun normalized(): Vec3f {
        val lenSq = lengthSquared()
        if (lenSq <= 1e-8f) return this
        val inv = 1f / sqrt(lenSq)
        return this * inv
    }
}

data class Aabb3(
    val min: Vec3f,
    val max: Vec3f,
) {
    val center: Vec3f get() = (min + max) * 0.5f
    val size: Vec3f get() = max - min

    fun contains(point: Vec3f): Boolean {
        return point.x >= min.x && point.x <= max.x &&
            point.y >= min.y && point.y <= max.y &&
            point.z >= min.z && point.z <= max.z
    }
}

data class DetectedPlane(
    val type: PlaneType,
    val normal: Vec3f,
    val pointOnPlane: Vec3f,
) {
    enum class PlaneType {
        FLOOR,
        CEILING,
        WALL,
    }

    fun distanceTo(point: Vec3f): Float = normal.dot(point - pointOnPlane)
}

data class FloorUvBounds(
    val min: Vec2f,
    val max: Vec2f,
)

data class FreeFloorRegion(
    val polygon: List<Vec2f>,
    val areaSqM: Float,
    val uvBounds: FloorUvBounds,
    val occupancyRatio: Float? = null,
) {
    fun fits(width: Float, depth: Float): Boolean {
        val regionWidth = uvBounds.max.x - uvBounds.min.x
        val regionDepth = uvBounds.max.y - uvBounds.min.y
        return (width <= regionWidth && depth <= regionDepth) ||
            (depth <= regionWidth && width <= regionDepth)
    }
}

data class RoomCorner(
    val position: Vec3f,
    val uv: Vec2f,
)

data class SurfacePalette(
    val floor: SurfaceColors?,
    val walls: SurfaceColors?,
    val ceiling: SurfaceColors?,
) {
    enum class MaterialHint {
        WOOD,
        TILE,
        CARPET,
        CONCRETE,
        BRICK,
        PLASTER,
        MARBLE,
        UNKNOWN,
    }

    data class SurfaceColors(
        val primary: Vec3f,
        val secondary: Vec3f? = null,
        val hint: MaterialHint = MaterialHint.UNKNOWN,
    ) {
        val dominantColors: List<Vec3f>
            get() = listOfNotNull(primary, secondary)
    }

    companion object {
        val EMPTY = SurfacePalette(floor = null, walls = null, ceiling = null)
    }
}

data class SourceCameraInfo(
    val focalLengthMm: Float? = null,
    val sensorWidthMm: Float? = null,
    val focalLength35mmEquivalentMm: Float? = null,
    val subjectDistanceMeters: Float? = null,
    val imageWidthPx: Int? = null,
    val imageHeightPx: Int? = null,
    val photoOrientation: Int? = null,
)

data class RoomModel(
    val aabb: Aabb3,
    val floor: DetectedPlane,
    val ceiling: DetectedPlane?,
    val walls: List<DetectedPlane>,
    val corners: List<RoomCorner>,
    val freeFloorRegions: List<FreeFloorRegion>,
    val surfacePalette: SurfacePalette,
    val cameraInfo: SourceCameraInfo?,
    val sceneToMeters: Float,
) {
    val roomBounds: Aabb3 get() = aabb

    val interiorHeightSceneUnits: Float?
        get() = ceiling?.let { abs(it.distanceTo(floor.pointOnPlane)) }?.takeIf { it > 0.001f }

    val interiorFootprintSceneUnits: Pair<Float, Float>
        get() {
            var widthScene = aabb.size.x
            var depthScene = aabb.size.z
            var bestWidthSpan = 0f
            var bestDepthSpan = 0f
            val opposingDotThreshold = -0.45f
            val minimumUsableSpan = 0.05f
            for (i in walls.indices) {
                for (j in (i + 1) until walls.size) {
                    val first = walls[i]
                    val second = walls[j]
                    if (first.normal.dot(second.normal) >= opposingDotThreshold) continue
                    val separation = abs(second.distanceTo(first.pointOnPlane))
                    if (separation <= 0.001f) continue
                    val ax = abs(first.normal.x)
                    val az = abs(first.normal.z)
                    if (max(ax, az) <= 0.05f) continue
                    if (ax >= az) {
                        bestWidthSpan = max(bestWidthSpan, separation)
                    } else {
                        bestDepthSpan = max(bestDepthSpan, separation)
                    }
                }
            }
            if (bestWidthSpan > minimumUsableSpan) widthScene = bestWidthSpan
            if (bestDepthSpan > minimumUsableSpan) depthScene = bestDepthSpan
            return widthScene to depthScene
        }

    val widthMeters: Float get() = interiorFootprintSceneUnits.first * sceneToMeters
    val depthMeters: Float get() = interiorFootprintSceneUnits.second * sceneToMeters
    val heightMeters: Float get() = (interiorHeightSceneUnits ?: aabb.size.y) * sceneToMeters
}

data class RoomFurnitureDimensions(
    val widthM: Float,
    val heightM: Float,
    val depthM: Float,
)

data class ClearanceReport(
    val frontM: Float?,
    val backM: Float?,
    val leftM: Float?,
    val rightM: Float?,
) {
    val warnings: List<String>
        get() = buildList {
            if (frontM != null && frontM < MINIMUM_PASSAGE_M) add("Front clearance below 0.6m.")
            if (leftM != null && leftM < MINIMUM_PASSAGE_M) add("Left clearance below 0.6m.")
            if (rightM != null && rightM < MINIMUM_PASSAGE_M) add("Right clearance below 0.6m.")
        }

    companion object {
        const val MINIMUM_PASSAGE_M = 0.6f
    }
}

data class FitLocation(
    val centerScene: Vec3f,
    val yRotationRad: Float,
    val clearance: ClearanceReport,
    val regionIndex: Int,
)

data class FitCheckResult(
    val fitsInRoom: Boolean,
    val fitLocations: List<FitLocation>,
    val warnings: List<String>,
)

class FitCheckEngine(
    private val roomModel: RoomModel,
) {
    fun checkFit(furniture: RoomFurnitureDimensions): FitCheckResult {
        val roomWidth = roomModel.widthMeters
        val roomDepth = roomModel.depthMeters
        val roomHeight = roomModel.heightMeters
        val widthFits = furniture.widthM <= roomWidth || furniture.depthM <= roomWidth
        val depthFits = furniture.depthM <= roomDepth || furniture.widthM <= roomDepth
        val heightFits = furniture.heightM <= roomHeight

        if (!widthFits || !depthFits) {
            return FitCheckResult(
                fitsInRoom = false,
                fitLocations = emptyList(),
                warnings = listOf("Furniture footprint exceeds room extents."),
            )
        }

        val warnings = mutableListOf<String>()
        if (!heightFits) {
            warnings += "Furniture height may exceed room height."
        }

        val locations = roomModel.freeFloorRegions.mapIndexedNotNull { index, region ->
            val widthFitsRegion = region.fits(furniture.widthM, furniture.depthM)
            val rotatedFitsRegion = region.fits(furniture.depthM, furniture.widthM)
            if (!widthFitsRegion && !rotatedFitsRegion) return@mapIndexedNotNull null

            val centerUv = (region.uvBounds.min + region.uvBounds.max) * 0.5f
            val floorOrigin = roomModel.floor.pointOnPlane
            val centerScene = Vec3f(
                floorOrigin.x + centerUv.x,
                floorOrigin.y,
                floorOrigin.z + centerUv.y,
            )
            val regionSpan = sqrt(max(region.areaSqM, 0f))
            val clearance = ClearanceReport(
                frontM = max(0f, regionSpan - furniture.depthM),
                backM = null,
                leftM = max(0f, regionSpan - furniture.widthM),
                rightM = null,
            )
            FitLocation(
                centerScene = centerScene,
                yRotationRad = if (widthFitsRegion) 0f else (Math.PI.toFloat() / 2f),
                clearance = clearance,
                regionIndex = index,
            )
        }

        warnings += locations.flatMap { it.clearance.warnings }
        return FitCheckResult(fitsInRoom = true, fitLocations = locations, warnings = warnings)
    }
}

data class CornerPlacementSuggestion(
    val corner: RoomCorner,
    val suggestedPositionScene: Vec3f,
    val yRotationRad: Float,
    val clearance: ClearanceReport,
    val score: Float,
    val rationale: String,
)

class CornerPlacement(
    private val roomModel: RoomModel,
) {
    fun suggestions(furniture: RoomFurnitureDimensions): List<CornerPlacementSuggestion> {
        return roomModel.corners.flatMap { corner ->
            candidateRotations(corner).mapNotNull { rotation ->
                evaluate(furniture, corner, rotation)
            }
        }.sortedByDescending { it.score }
    }

    private fun candidateRotations(corner: RoomCorner): List<Float> {
        val base = inwardBisectorYaw(corner)
        return listOf(base, base + (Math.PI.toFloat() / 2f))
    }

    private fun inwardBisectorXz(corner: RoomCorner): Vec3f {
        val center = roomModel.aabb.center
        val flatCenter = Vec3f(center.x, roomModel.floor.pointOnPlane.y, center.z)
        val delta = Vec3f(
            flatCenter.x - corner.position.x,
            0f,
            flatCenter.z - corner.position.z,
        )
        return if (delta.lengthSquared() < 1e-8f) Vec3f(1f, 0f, 0f) else delta.normalized()
    }

    private fun inwardBisectorYaw(corner: RoomCorner): Float {
        val bisector = inwardBisectorXz(corner)
        return atan2(bisector.z, bisector.x)
    }

    private fun evaluate(
        furniture: RoomFurnitureDimensions,
        corner: RoomCorner,
        rotation: Float,
    ): CornerPlacementSuggestion? {
        val sceneScale = max(roomModel.sceneToMeters, 0.0001f)
        val halfDiagonalScene = sqrt(
            furniture.widthM * furniture.widthM + furniture.depthM * furniture.depthM,
        ) / (2f * sceneScale)
        val bisector = inwardBisectorXz(corner)
        val position = corner.position + (bisector * halfDiagonalScene)
        if (!roomModel.roomBounds.contains(position)) return null

        val clearance = ClearanceReport(frontM = 1.0f, backM = 0.2f, leftM = 0.2f, rightM = 0.2f)
        val score = min(1f, (clearance.frontM ?: 0f) / 2f) * 0.7f + if (roomModel.corners.size > 1) 0.3f else 0.1f
        return CornerPlacementSuggestion(
            corner = corner,
            suggestedPositionScene = position,
            yRotationRad = rotation,
            clearance = clearance,
            score = score,
            rationale = "Corner placement scored from front clearance and corner confidence.",
        )
    }
}
