package com.furnit.android.roomreconstruction

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import com.furnit.android.services.RoomMeasurementDisplay
import com.furnit.android.utils.LogUtil
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.TimeUnit
import kotlin.math.max

data class RoomMeasurementPipelineResult(
    val width: Float,
    val height: Float,
    val depth: Float,
    val measured: Boolean,
    val source: String,
    val focalSource: String,
)

object DepthAnythingRoomMeasurementPipeline {
    private const val TAG = "RoomMeasurementPipeline"

    fun measure(
        context: Context,
        workingImage: Bitmap,
        rawDepth: FloatArray,
        imageUri: Uri? = null,
        cameraMetadata: Map<String, Double>? = null,
    ): RoomMeasurementPipelineResult {
        val imageWidth = workingImage.width
        val imageHeight = workingImage.height
        require(rawDepth.size == imageWidth * imageHeight)

        val executor = Executors.newFixedThreadPool(2)
        return try {
            val geoCalibFuture: Future<GeoCalibCalibrationResult?> = executor.submit(
                Callable { GeoCalibCalibrationService.estimateCalibration(context, workingImage) },
            )
            val objectRectFuture: Future<MeasurementObjectRect?> = executor.submit(
                Callable { MeasurementObjectDetection.detectMeasurementObject(context, workingImage) },
            )

            val geoCalib = geoCalibFuture.get(60, TimeUnit.SECONDS)
            val objectRect = objectRectFuture.get(60, TimeUnit.SECONDS)

            val exifFocalPx = readExifFocalPixels(context, imageUri, imageWidth)
            val geoFocalPx = exifFocalPx ?: FocalResolver.resolve(
                vpFocalPx = null,
                geoCalib = geoCalib,
                imageWidth = imageWidth,
                imageHeight = imageHeight,
            ).fx

            val rawObjectMeasured = MeasurementObjectDetection.measureObjectBBox(
                objectRect = objectRect,
                depth = rawDepth,
                imageWidth = imageWidth,
                imageHeight = imageHeight,
                fx = geoFocalPx,
                fy = geoFocalPx,
            )
            val metricCalibration = resolveMetricCalibration(
                geoFocalPx = geoFocalPx,
                exifFocalPx = exifFocalPx,
                rawObjectMeasurement = rawObjectMeasured,
            )
            val focalPx = metricCalibration.focalPx

            val measurementFocal = measurementFocalPixels(
                geoCalib = geoCalib,
                imageWidth = imageWidth,
                imageHeight = imageHeight,
                fallbackFx = focalPx,
            )
            val depthRows = RoomMath.depthToRows(rawDepth, imageWidth, imageHeight)
            val arkitGravityDown = arkitGravityDownVector(cameraMetadata)
            val arkitCameraHeight = cameraMetadata?.get("arkitCameraHeightM")?.toFloat()
            val furnitureExcludeBBox = rawObjectMeasured?.let {
                FurnitureExcludeBBox(it.bboxLeftX, it.bboxRightX, it.bboxTopY, it.bboxBottomY)
            }
            val measurementCalibration = MeasurementDepthCalibrationResolver.resolve(
                rawDepthFlat = rawDepth,
                geoCalib = geoCalib,
                arkitGravityDown = arkitGravityDown,
                arkitCameraHeightMeters = arkitCameraHeight,
                imageWidth = imageWidth,
                imageHeight = imageHeight,
                fx = measurementFocal.fx,
                fy = measurementFocal.fy,
                wallMargin = RoomMeasurementConstants.WALL_MARGIN,
                furnitureExcludeBBox = furnitureExcludeBBox,
                objectRect = objectRect,
                depthRows = depthRows,
            )
            val measurementDepth = RoomMath.scaleDepthFlat(rawDepth, measurementCalibration.depthScale)

            val wallMeasured = WallMeasurementMeasure.measureWall(
                depth = measurementDepth,
                imageWidth = imageWidth,
                imageHeight = imageHeight,
                fx = measurementFocal.fx,
                fy = measurementFocal.fy,
                wallMargin = RoomMeasurementConstants.WALL_MARGIN,
            )
            val depthSpreadMeasured = DepthSpreadMeasure.measureDepthSpread(
                pointGrid = measurementCalibration.pointGrid,
                imageWidth = imageWidth,
                imageHeight = imageHeight,
                wallMargin = RoomMeasurementConstants.WALL_MARGIN,
                scale = measurementCalibration.depthScale,
                cameraHeightPriorMeters = measurementCalibration.cameraHeightPriorMeters,
            )?.let { WallMeasurement(it.width, it.height, it.depth) } ?: wallMeasured

            val detections = objectRect?.let {
                listOf(
                    ObjectDetectionBox(
                        cls = it.classIdx,
                        box = intArrayOf(it.leftX, it.topY, it.rightX, it.bottomY),
                        conf = it.confidence,
                    ),
                )
            } ?: emptyList()
            val depthMask = RoomExtentMeasure.buildInvalidDepthMask(
                depth = rawDepth,
                width = imageWidth,
                height = imageHeight,
                detections = detections,
                focalPx = measurementFocal.fx,
                cx = (imageWidth - 1) * 0.5f,
                cy = (imageHeight - 1) * 0.5f,
            )
            val roomExtentPoints = RoomExtentMeasure.roomExtentPoints(
                pointGrid = measurementCalibration.pointGrid,
                width = imageWidth,
                height = imageHeight,
                valid = depthMask.valid,
                stride = 2,
            )
            val roomExtentMeasured = RoomExtentMeasure.roomExtentFromWalls(
                points = roomExtentPoints,
                scale = measurementCalibration.depthScale,
                cameraHeight = measurementCalibration.cameraHeightPriorMeters,
            )

            val gravityPitch = MeasurementDepthCalibrationResolver.measurementPitchRadians(
                geoCalib = geoCalib,
                gravitySourceCode = measurementCalibration.gravitySourceCode,
                arkitGravityDown = arkitGravityDown,
            )
            val roomHeightMeasured = RoomHeightMeasure.roomHeightSingleView(
                pointGrid = measurementCalibration.pointGrid,
                fy = measurementFocal.fy,
                cy = (imageHeight - 1) * 0.5f,
                pitch = gravityPitch,
                cameraHeight = measurementCalibration.cameraHeightPriorMeters,
            )
            val roomWidthMeasured = RoomHeightMeasure.roomWidthSingleView(
                pointGrid = measurementCalibration.pointGrid,
                vFloor = roomHeightMeasured.vFloor,
                vHorizon = roomHeightMeasured.vHorizon,
                vCeil = roomHeightMeasured.vCeil,
                normalSign = roomHeightMeasured.normalSign,
                cameraHeight = measurementCalibration.cameraHeightPriorMeters,
            )

            val authoritativeHeight = if (roomHeightMeasured.confidence >= 0.5f) {
                roomHeightMeasured.height
            } else {
                roomExtentMeasured.height.coerceIn(2.0f, 3.6f)
            }
            val authoritativeWidth = if (roomWidthMeasured.confidence >= 0.5f) {
                roomWidthMeasured.width
            } else {
                roomExtentMeasured.width
            }
            val measured = if (roomExtentMeasured.width > 0f && roomExtentMeasured.height > 0f && roomExtentMeasured.depth > 0f) {
                WallMeasurement(authoritativeWidth, authoritativeHeight, roomExtentMeasured.depth)
            } else {
                WallMeasurementMeasure.sanitizeRoomMeasurement(depthSpreadMeasured, wallMeasured)
            }

            val meshWidth = RoomMeasurementDisplay.meshRoomWidthMeters(
                measured.width,
                imageWidth,
                imageHeight,
            )
            LogUtil.i(
                TAG,
                "[measure] ${imageWidth}x$imageHeight focal=${measurementFocal.source} " +
                    "geocalib=${geoCalib != null} rtmdet=${objectRect != null} " +
                    "Hconf=${"%.2f".format(roomHeightMeasured.confidence)} " +
                    "Wconf=${"%.2f".format(roomWidthMeasured.confidence)} " +
                    "extentConf=${"%.2f".format(roomExtentMeasured.confidence)} " +
                    "scale=${"%.4f".format(measurementCalibration.depthScale)} " +
                    "W=${measured.width} H=${measured.height} D=${measured.depth} meshW=$meshWidth",
            )
            RoomMeasurementPipelineResult(
                width = meshWidth,
                height = measured.height,
                depth = measured.depth,
                measured = true,
                source = "depth_anything_metric_ios_pipeline",
                focalSource = measurementFocal.source,
            )
        } catch (error: Exception) {
            LogUtil.e(TAG, "Pipeline measurement failed", error)
            throw error
        } finally {
            executor.shutdownNow()
        }
    }

    private data class DepthMetricCalibration(
        val depthScale: Float,
        val focalPx: Float,
        val sourceLabel: String,
    )

    private fun resolveMetricCalibration(
        geoFocalPx: Float,
        exifFocalPx: Float?,
        rawObjectMeasurement: ObjectBBoxMeasurement?,
    ): DepthMetricCalibration {
        var focalPx = geoFocalPx
        var depthScale = 1.0f
        var sourceLabel = "unchanged"
        val anchorExpectedHeight = rawObjectMeasurement?.classIdx?.let {
            RoomMeasurementConstants.objectAnchorHeightMeters[it]
        }
        val anchorMeasuredHeight = rawObjectMeasurement?.height
        val anchorDepthScale = if (anchorExpectedHeight != null && anchorMeasuredHeight != null && anchorMeasuredHeight > 0.2f) {
            (anchorExpectedHeight / anchorMeasuredHeight).coerceIn(
                RoomMeasurementConstants.DEPTH_METRIC_SCALE_MIN,
                RoomMeasurementConstants.DEPTH_METRIC_SCALE_MAX,
            )
        } else {
            null
        }

        if (exifFocalPx != null && exifFocalPx > 1f) {
            val focalRatio = geoFocalPx / exifFocalPx
            if (focalRatio in RoomMeasurementConstants.GEO_EXIF_FOCAL_MATCH_RATIO_MIN..RoomMeasurementConstants.GEO_EXIF_FOCAL_MATCH_RATIO_MAX) {
                if (anchorDepthScale != null) {
                    depthScale = anchorDepthScale
                    sourceLabel = "depth_anchor_exif_confirms_focal"
                } else {
                    sourceLabel = "exif_confirms_focal_no_anchor"
                }
            } else if (exifFocalPx > geoFocalPx * RoomMeasurementConstants.GEO_EXIF_FOCAL_MATCH_RATIO_MAX) {
                focalPx = exifFocalPx
                sourceLabel = "exif_focal_override"
            } else if (anchorDepthScale != null) {
                depthScale = anchorDepthScale
                sourceLabel = "depth_anchor_focal_mismatch"
            }
        } else if (anchorDepthScale != null) {
            depthScale = anchorDepthScale
            sourceLabel = "depth_anchor_no_exif"
        }
        return DepthMetricCalibration(depthScale, focalPx, sourceLabel)
    }

    private fun measurementFocalPixels(
        geoCalib: GeoCalibCalibrationResult?,
        imageWidth: Int,
        imageHeight: Int,
        fallbackFx: Float,
    ): ResolvedFocal {
        return FocalResolver.resolve(null, geoCalib, imageWidth, imageHeight).let { resolved ->
            if (resolved.fx > 1f) resolved else ResolvedFocal(fallbackFx, fallbackFx, "fallback_metric_focal", 70f, false)
        }
    }

    private fun arkitGravityDownVector(metadata: Map<String, Double>?): Vec3? {
        if (metadata == null) return null
        val x = metadata["arkitGravityDownImageX"] ?: return null
        val y = metadata["arkitGravityDownImageY"] ?: return null
        val z = metadata["arkitGravityDownImageZ"] ?: return null
        val vector = Vec3(x.toFloat(), y.toFloat(), z.toFloat())
        if (!vector.x.isFinite() || !vector.y.isFinite() || !vector.z.isFinite()) return null
        if (vector.length() <= 0.5f) return null
        return vector.normalized()
    }

    private fun readExifFocalPixels(context: Context, imageUri: Uri?, imageWidth: Int): Float? {
        if (imageUri == null) return null
        return try {
            context.contentResolver.openInputStream(imageUri)?.use { stream ->
                val exif = ExifInterface(stream)
                val raw = exif.getAttribute(ExifInterface.TAG_FOCAL_LENGTH_IN_35MM_FILM)
                    ?: exif.getAttribute("FocalLengthIn35mmFilm")
                val focal35 = raw?.toFloatOrNull()?.takeIf { it > 1f } ?: return null
                (focal35 / 36.0f) * imageWidth
            }
        } catch (_: Exception) {
            null
        }
    }
}
