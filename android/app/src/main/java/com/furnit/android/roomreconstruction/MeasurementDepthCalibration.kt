package com.furnit.android.roomreconstruction

import kotlin.math.atan2
import kotlin.math.roundToInt

data class MeasurementDepthCalibration(
    val depthScale: Float,
    val cameraHeightPriorMeters: Float,
    val cameraHeightRawMeters: Float?,
    val sourceLabel: String,
    val sourceCode: Int,
    val levelingRotation: Mat3,
    val pointGrid: LeveledDepthPointGrid,
    val gravitySourceCode: Int,
    val cameraHeightPriorSourceCode: Int,
    val scaleEstimatorConfidence: Float,
    val scaleEstimatorDebug: String,
)

data class FurnitureExcludeBBox(val leftX: Int, val rightX: Int, val topY: Int, val bottomY: Int)

object MeasurementDepthCalibrationResolver {
    fun resolve(
        rawDepthFlat: FloatArray,
        geoCalib: GeoCalibCalibrationResult?,
        arkitGravityDown: Vec3?,
        arkitCameraHeightMeters: Float?,
        imageWidth: Int,
        imageHeight: Int,
        fx: Float,
        fy: Float,
        wallMargin: Float,
        furnitureExcludeBBox: FurnitureExcludeBBox?,
        objectRect: MeasurementObjectRect?,
        depthRows: RoomDepthMap,
    ): MeasurementDepthCalibration {
        var cameraHeightPrior: Float
        var priorSourceCode: Int
        if (arkitCameraHeightMeters != null && arkitCameraHeightMeters in 0.5f..2.2f) {
            cameraHeightPrior = arkitCameraHeightMeters
            priorSourceCode = 1
        } else {
            cameraHeightPrior = RoomMeasurementConstants.CAMERA_HEIGHT_PRIOR_METERS
            priorSourceCode = 0
        }

        val levelingRotation: Mat3
        val gravitySourceCode: Int
        if (arkitGravityDown != null) {
            levelingRotation = GeoCalibCalibrationResult.levelingRotationMatrix(
                Vec3(-arkitGravityDown.x, -arkitGravityDown.y, -arkitGravityDown.z),
            )
            gravitySourceCode = 2
        } else if (geoCalib != null &&
            kotlin.math.abs(geoCalib.rollRadians) <= RoomMeasurementConstants.MAX_PLAUSIBLE_ROLL_RADIANS &&
            kotlin.math.abs(geoCalib.pitchRadians) <= RoomMeasurementConstants.MAX_PLAUSIBLE_PITCH_RADIANS
        ) {
            levelingRotation = geoCalib.levelingRotationMatrix()
            gravitySourceCode = 1
        } else {
            levelingRotation = Mat3.identity()
            gravitySourceCode = 0
        }

        val cx = (imageWidth - 1) * 0.5f
        val cy = (imageHeight - 1) * 0.5f
        val pointGrid = LeveledDepthPointGrid(
            depth = rawDepthFlat,
            width = imageWidth,
            height = imageHeight,
            fx = fx,
            fy = fy,
            cx = cx,
            cy = cy,
            rotation = levelingRotation,
        )
        val rawCameraHeight = cameraHeightFromFloorSamples(
            pointGrid = pointGrid,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            wallMargin = wallMargin,
            furnitureExcludeBBox = furnitureExcludeBBox,
        )

        var depthScale = 1.0f
        var sourceLabel = "camera_height_unavailable"
        var sourceCode = 20
        if (rawCameraHeight != null &&
            rawCameraHeight in RoomMeasurementConstants.CAMERA_HEIGHT_RAW_VALID_MIN..RoomMeasurementConstants.CAMERA_HEIGHT_RAW_VALID_MAX
        ) {
            depthScale = cameraHeightPrior / rawCameraHeight
            sourceLabel = if (priorSourceCode == 1) {
                "camera_height_arkit_floor_plane"
            } else {
                "camera_height_prior_1.7m"
            }
            sourceCode = 21
        }

        var scaleEstimatorConfidence = 0f
        var scaleEstimatorDebug = "not_run"
        if (priorSourceCode == 0) {
            val vpGravity = VanishingPointGravity.refine(levelingRotation, fx.toDouble())
            val scaleEstimatorPointGrid = LeveledDepthPointGrid(
                depth = rawDepthFlat,
                width = imageWidth,
                height = imageHeight,
                fx = fx,
                fy = fy,
                cx = cx,
                cy = cy,
                rotation = vpGravity.levelingRotation,
            )
            val objectBoxes = objectRect?.let { rect ->
                listOf(
                    ScaleObjectBox(
                        classIdx = rect.classIdx,
                        confidence = rect.confidence,
                        leftX = rect.leftX.toFloat(),
                        topY = rect.topY.toFloat(),
                        rightX = rect.rightX.toFloat(),
                        bottomY = rect.bottomY.toFloat(),
                    ),
                )
            } ?: emptyList()
            val furnitureBoxes = furnitureExcludeBBox?.let { bbox ->
                listOf(floatArrayOf(bbox.leftX.toFloat(), bbox.topY.toFloat(), bbox.rightX.toFloat(), bbox.bottomY.toFloat()))
            } ?: emptyList()
            val ctx = SceneContext(
                rawDepth = depthRows,
                focalPx = fx.toDouble(),
                levelingRotation = vpGravity.levelingRotation,
                furnitureBoxes = furnitureBoxes,
                objectBoxes = objectBoxes,
                imageWidth = imageWidth,
                imageHeight = imageHeight,
                rawCameraHeightMeters = rawCameraHeight?.toDouble(),
                fallbackDepthScale = depthScale.toDouble(),
                impliedRoomHeightForScale = impliedRoomHeightForScale@ { candidateScale ->
                    if (!candidateScale.isFinite() || candidateScale <= 0.0) return@impliedRoomHeightForScale null
                    val impliedCameraHeightPrior = rawCameraHeight?.let { it * candidateScale.toFloat() } ?: cameraHeightPrior
                    DepthSpreadMeasure.measureDepthSpread(
                        pointGrid = scaleEstimatorPointGrid,
                        imageWidth = imageWidth,
                        imageHeight = imageHeight,
                        wallMargin = wallMargin,
                        scale = candidateScale.toFloat(),
                        cameraHeightPriorMeters = impliedCameraHeightPrior,
                    )?.height?.toDouble()
                },
            )
            val estimatorResult = ScaleEstimator().estimate(ctx)
            depthScale = estimatorResult.depthScale.toFloat()
            scaleEstimatorConfidence = estimatorResult.confidence.toFloat()
            scaleEstimatorDebug = estimatorResult.source
            if (rawCameraHeight != null && scaleEstimatorConfidence > 0.05f) {
                cameraHeightPrior = rawCameraHeight * depthScale
                priorSourceCode = 2
            }
            sourceLabel = estimatorResult.source
            sourceCode = 22
        }

        return MeasurementDepthCalibration(
            depthScale = depthScale,
            cameraHeightPriorMeters = cameraHeightPrior,
            cameraHeightRawMeters = rawCameraHeight,
            sourceLabel = sourceLabel,
            sourceCode = sourceCode,
            levelingRotation = levelingRotation,
            pointGrid = pointGrid,
            gravitySourceCode = gravitySourceCode,
            cameraHeightPriorSourceCode = priorSourceCode,
            scaleEstimatorConfidence = scaleEstimatorConfidence,
            scaleEstimatorDebug = scaleEstimatorDebug,
        )
    }

    fun measurementPitchRadians(
        geoCalib: GeoCalibCalibrationResult?,
        gravitySourceCode: Int,
        arkitGravityDown: Vec3?,
    ): Float {
        if (gravitySourceCode == 2 && arkitGravityDown != null) {
            val down = arkitGravityDown.normalized()
            if (!down.y.isFinite() || !down.z.isFinite() || kotlin.math.abs(down.y) <= 0.1f) return 0f
            val pitch = atan2(-down.z, down.y)
            return if (kotlin.math.abs(pitch) <= RoomMeasurementConstants.MAX_PLAUSIBLE_PITCH_RADIANS) pitch else 0f
        }
        if (geoCalib == null || gravitySourceCode != 1 ||
            kotlin.math.abs(geoCalib.pitchRadians) > RoomMeasurementConstants.MAX_PLAUSIBLE_PITCH_RADIANS
        ) {
            return 0f
        }
        return geoCalib.pitchRadians
    }

    private fun cameraHeightFromFloorSamples(
        pointGrid: LeveledDepthPointGrid,
        imageWidth: Int,
        imageHeight: Int,
        wallMargin: Float,
        furnitureExcludeBBox: FurnitureExcludeBBox?,
    ): Float? {
        if (imageWidth <= 1 || imageHeight <= 1) return null
        val margin = wallMargin.coerceIn(0f, 0.45f)
        val leftX = (margin * imageWidth).roundToInt()
        val rightX = ((1f - margin) * imageWidth).roundToInt() - 1
        val floorStartY = (imageHeight * RoomMeasurementConstants.FLOOR_BAND_START_FRACTION).roundToInt()
        val bottomY = imageHeight - 1
        if (leftX >= rightX || floorStartY >= bottomY) return null

        val step = maxOf(4, (bottomY - floorStartY) / 32)
        val bandWidth = maxOf((rightX - leftX).toFloat(), 1f)
        val bandHeight = maxOf((bottomY - floorStartY).toFloat(), 1f)

        val excludeRect = furnitureExcludeBBox?.let { bbox ->
            val padX = ((bbox.rightX - bbox.leftX) * 0.05f).roundToInt() + 2
            val padY = ((bbox.bottomY - bbox.topY) * 0.05f).roundToInt() + 2
            FurnitureExcludeBBox(
                leftX = maxOf(0, bbox.leftX - padX),
                rightX = minOf(imageWidth - 1, bbox.rightX + padX),
                topY = maxOf(0, bbox.topY - padY),
                bottomY = minOf(imageHeight - 1, bbox.bottomY + padY),
            )
        }

        fun isExcluded(row: Int, column: Int): Boolean {
            if (excludeRect != null) {
                return column in excludeRect.leftX..excludeRect.rightX &&
                    row in excludeRect.topY..excludeRect.bottomY
            }
            val u = (column - leftX) / bandWidth
            val v = (row - floorStartY) / bandHeight
            return u > RoomMeasurementConstants.FLOOR_CHAIR_EXCLUDE_U &&
                v > RoomMeasurementConstants.FLOOR_CHAIR_EXCLUDE_V
        }

        val cameraHeights = ArrayList<Float>(512)
        var row = floorStartY
        while (row <= bottomY) {
            var column = leftX
            while (column <= rightX) {
                if (!isExcluded(row, column)) {
                    val leveledPoint = pointGrid.point(column, row)
                    if (leveledPoint != null && leveledPoint.y > 0.05f) {
                        cameraHeights += leveledPoint.y
                    }
                }
                column += step
            }
            row += step
        }
        if (cameraHeights.size < 32) return null
        return RoomMath.median(cameraHeights)
    }
}
