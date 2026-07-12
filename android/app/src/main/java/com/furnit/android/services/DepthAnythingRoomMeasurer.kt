package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import com.furnit.android.RoomDefaults
import com.furnit.android.utils.LogUtil
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Depth Anything metric indoor depth + pinhole wall projection (Swift/Python parity without GeoCalib).
 *
 * Uses EXIF 35mm-equivalent focal when available; otherwise 28mm fallback. Applies a 1.7m camera-height
 * prior from the floor band when plausible, then measures W×H×D from the depth map.
 */
object DepthAnythingRoomMeasurer {
    private const val TAG = "DepthAnythingRoomMeasure"
    private const val MODEL_INPUT_SIZE = 518
    private const val MAX_WORKING_IMAGE_DIMENSION = 1600
    private const val WALL_MARGIN = 0.05f
    private const val CAMERA_HEIGHT_PRIOR_METERS = 1.70f
    private const val CAMERA_HEIGHT_RAW_VALID_MIN = 0.45f
    private const val CAMERA_HEIGHT_RAW_VALID_MAX = 5.0f
    private const val FLOOR_BAND_START_FRACTION = 0.78f
    private const val FLOOR_CHAIR_EXCLUDE_U = 0.58f
    private const val FLOOR_CHAIR_EXCLUDE_V = 0.55f
    private const val PLAUSIBLE_SPAN_MIN = 1.2f
    private const val PLAUSIBLE_SPAN_MAX = 8.0f
    private const val FALLBACK_FOCAL_35MM = 28.0f

    private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    data class Result(
        val width: Float,
        val height: Float,
        val depth: Float,
        val measured: Boolean,
        val source: String,
        val focalSource: String = "unknown",
    ) {
        fun toRoomDimensions(): SinglePhotoRoomReconstructor.RoomDimensions {
            return SinglePhotoRoomReconstructor.RoomDimensions(
                width = width,
                depth = depth,
                height = height,
            )
        }
    }

    private data class DepthSession(
        val env: OrtEnvironment,
        val session: OrtSession,
        val inputName: String,
        val outputName: String,
    )

    private val sessionLock = Any()
    @Volatile private var cachedSession: DepthSession? = null
    private val inferenceLock = Any()

    fun measure(
        context: Context,
        image: Bitmap,
        imageUri: Uri? = null,
    ): Result {
        val oriented = orientBitmap(image)
        val working = downsample(oriented, MAX_WORKING_IMAGE_DIMENSION)
        if (working !== oriented && oriented !== image) oriented.recycle()

        return try {
            val depth = runDepthInference(context, working) ?: return fallback("depth_inference_failed")
            val imageWidth = working.width
            val imageHeight = working.height
            val (fx, fy, focalSource) = focalPixels(context, imageUri, imageWidth, imageHeight)

            val scaledDepth = applyCameraHeightPriorScale(depth, imageWidth, imageHeight)
            val wall = measureWall(scaledDepth, imageWidth, imageHeight, fx, fy)
            val spreadDepth = measureDepthSpread(scaledDepth, imageWidth, imageHeight) ?: wall.depth
            val sanitized = sanitizeMeasurement(
                width = wall.width,
                height = wall.height,
                depth = spreadDepth,
                wallFallback = wall,
            )
            val meshWidth = RoomMeasurementDisplay.meshRoomWidthMeters(sanitized.width, imageWidth, imageHeight)
            LogUtil.i(
                TAG,
                "[measure] ${imageWidth}x$imageHeight focal=$focalSource " +
                    "W=${sanitized.width} H=${sanitized.height} D=${sanitized.depth} meshW=$meshWidth",
            )
            Result(
                width = meshWidth,
                height = sanitized.height,
                depth = sanitized.depth,
                measured = true,
                source = "depth_anything_metric",
                focalSource = focalSource,
            )
        } catch (error: Exception) {
            LogUtil.e(TAG, "Room measurement failed", error)
            fallback("exception:${error.javaClass.simpleName}")
        } finally {
            if (working !== image) working.recycle()
        }
    }

    fun measureFromFile(context: Context, imageFile: File): Result {
        val bitmap = android.graphics.BitmapFactory.decodeFile(imageFile.absolutePath)
            ?: return fallback("decode_failed")
        return try {
            measure(context, bitmap)
        } finally {
            bitmap.recycle()
        }
    }

    private fun fallback(reason: String): Result {
        return Result(
            width = RoomDefaults.DEFAULT_WIDTH_M,
            height = RoomDefaults.DEFAULT_HEIGHT_M,
            depth = RoomDefaults.DEFAULT_DEPTH_M,
            measured = false,
            source = reason,
        )
    }

    private fun orientBitmap(bitmap: Bitmap): Bitmap {
        return bitmap
    }

    private fun downsample(bitmap: Bitmap, maxDimension: Int): Bitmap {
        val longest = max(bitmap.width, bitmap.height)
        if (longest <= maxDimension) return bitmap
        val scale = maxDimension.toFloat() / longest.toFloat()
        val targetW = max(1, (bitmap.width * scale).roundToInt())
        val targetH = max(1, (bitmap.height * scale).roundToInt())
        return Bitmap.createScaledBitmap(bitmap, targetW, targetH, true)
    }

    private fun session(context: Context): DepthSession? {
        cachedSession?.let { return it }
        synchronized(sessionLock) {
            cachedSession?.let { return it }
            return try {
                val modelFile = copyAssetToCache(
                    context,
                    RoomGenerationAssets.DEPTH_ANYTHING_METRIC_INDOOR_SMALL_ONNX,
                )
                val env = OrtEnvironment.getEnvironment()
                val options = SessionOptions().apply {
                    setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT)
                    setExecutionMode(SessionOptions.ExecutionMode.SEQUENTIAL)
                    setIntraOpNumThreads(2)
                    setInterOpNumThreads(1)
                }
                val ortSession = env.createSession(modelFile.absolutePath, options)
                val inputName = ortSession.inputInfo.keys.first()
                val outputName = ortSession.outputInfo.keys.first()
                DepthSession(env, ortSession, inputName, outputName).also { cachedSession = it }
            } catch (error: Exception) {
                LogUtil.e(TAG, "Failed to load Depth Anything ONNX", error)
                null
            }
        }
    }

    private fun runDepthInference(context: Context, bitmap: Bitmap): FloatArray? {
        val depthSession = session(context) ?: return null
        synchronized(inferenceLock) {
            val input = preprocess(bitmap) ?: return null
            var tensor: OnnxTensor? = null
            return try {
                tensor = OnnxTensor.createTensor(
                    depthSession.env,
                    FloatBuffer.wrap(input.data),
                    input.shape,
                )
                val outputs = depthSession.session.run(mapOf(depthSession.inputName to tensor))
                val raw = outputs.use { result ->
                    extractDepthValues(result[0].value)
                } ?: return null
                resizeDepthBilinear(raw, input.modelWidth, input.modelHeight, bitmap.width, bitmap.height)
            } catch (error: Exception) {
                LogUtil.e(TAG, "Depth ONNX inference failed", error)
                null
            } finally {
                tensor?.close()
            }
        }
    }

    private data class PreprocessResult(
        val data: FloatArray,
        val shape: LongArray,
        val modelWidth: Int,
        val modelHeight: Int,
    )

    private fun preprocess(bitmap: Bitmap): PreprocessResult? {
        val scaled = Bitmap.createScaledBitmap(bitmap, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true)
        val pixels = IntArray(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)
        scaled.getPixels(pixels, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        if (scaled !== bitmap) scaled.recycle()

        val data = FloatArray(3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)
        var offset = 0
        for (channel in 0 until 3) {
            val mean = IMAGENET_MEAN[channel]
            val std = IMAGENET_STD[channel]
            for (pixel in pixels) {
                val component = when (channel) {
                    0 -> (pixel shr 16) and 0xFF
                    1 -> (pixel shr 8) and 0xFF
                    else -> pixel and 0xFF
                }
                data[offset++] = (component / 255.0f - mean) / std
            }
        }
        return PreprocessResult(
            data = data,
            shape = longArrayOf(1, 3, MODEL_INPUT_SIZE.toLong(), MODEL_INPUT_SIZE.toLong()),
            modelWidth = MODEL_INPUT_SIZE,
            modelHeight = MODEL_INPUT_SIZE,
        )
    }

    private fun extractDepthValues(value: Any?): FloatArray? {
        return when (value) {
            is FloatArray -> value
            is Array<*> -> {
                if (value.isEmpty()) return null
                if (value[0] is Float) {
                    FloatArray(value.size) { index -> value[index] as Float }
                } else {
                    val flattened = ArrayList<Float>(value.size * value.size)
                    for (entry in value) {
                        val nested = extractDepthValues(entry) ?: return null
                        for (sample in nested) flattened += sample
                    }
                    flattened.toFloatArray()
                }
            }
            else -> null
        }
    }

    private fun resizeDepthBilinear(
        source: FloatArray,
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int,
    ): FloatArray {
        if (sourceWidth == targetWidth && sourceHeight == targetHeight && source.size == targetWidth * targetHeight) {
            return source
        }
        val output = FloatArray(targetWidth * targetHeight)
        for (y in 0 until targetHeight) {
            val srcY = y.toFloat() / max(targetHeight - 1, 1) * max(sourceHeight - 1, 1)
            val y0 = srcY.toInt().coerceIn(0, sourceHeight - 1)
            val y1 = min(y0 + 1, sourceHeight - 1)
            val yFrac = srcY - y0
            for (x in 0 until targetWidth) {
                val srcX = x.toFloat() / max(targetWidth - 1, 1) * max(sourceWidth - 1, 1)
                val x0 = srcX.toInt().coerceIn(0, sourceWidth - 1)
                val x1 = min(x0 + 1, sourceWidth - 1)
                val xFrac = srcX - x0
                val v00 = source[y0 * sourceWidth + x0]
                val v10 = source[y0 * sourceWidth + x1]
                val v01 = source[y1 * sourceWidth + x0]
                val v11 = source[y1 * sourceWidth + x1]
                val top = v00 + (v10 - v00) * xFrac
                val bottom = v01 + (v11 - v01) * xFrac
                output[y * targetWidth + x] = top + (bottom - top) * yFrac
            }
        }
        return output
    }

    private fun focalPixels(
        context: Context,
        imageUri: Uri?,
        imageWidth: Int,
        imageHeight: Int,
    ): Triple<Float, Float, String> {
        val exifFocal35 = readExifFocal35mm(context, imageUri)
        if (exifFocal35 != null && exifFocal35 > 1f) {
            val focalPx = (exifFocal35 / 36.0f) * imageWidth
            return Triple(focalPx, focalPx, "exif_35mm_equiv_${"%.1f".format(exifFocal35)}mm")
        }
        val focalPx = (FALLBACK_FOCAL_35MM / 36.0f) * imageWidth
        return Triple(focalPx, focalPx, "fallback_35mm_equiv_${FALLBACK_FOCAL_35MM}mm")
    }

    private fun readExifFocal35mm(context: Context, imageUri: Uri?): Float? {
        if (imageUri == null) return null
        return try {
            context.contentResolver.openInputStream(imageUri)?.use { stream ->
                val exif = ExifInterface(stream)
                val raw = exif.getAttribute(ExifInterface.TAG_FOCAL_LENGTH_IN_35MM_FILM)
                    ?: exif.getAttribute("FocalLengthIn35mmFilm")
                raw?.toFloatOrNull()?.takeIf { it > 1f }
            }
        } catch (_: Exception) {
            null
        }
    }

    private fun applyCameraHeightPriorScale(
        depth: FloatArray,
        imageWidth: Int,
        imageHeight: Int,
    ): FloatArray {
        val floorStartY = (imageHeight * FLOOR_BAND_START_FRACTION).roundToInt().coerceIn(0, imageHeight - 1)
        val samples = ArrayList<Float>(256)
        for (y in floorStartY until imageHeight) {
            val v = y.toFloat() / max(imageHeight - 1, 1)
            for (x in 0 until imageWidth) {
                val u = x.toFloat() / max(imageWidth - 1, 1)
                if (u > FLOOR_CHAIR_EXCLUDE_U && v > FLOOR_CHAIR_EXCLUDE_V) continue
                val value = depth[y * imageWidth + x]
                if (value.isFinite() && value > 0f) samples += value
            }
        }
        if (samples.size < 24) return depth
        samples.sort()
        val cameraHeightRaw = samples[samples.size / 2]
        if (!cameraHeightRaw.isFinite() ||
            cameraHeightRaw !in CAMERA_HEIGHT_RAW_VALID_MIN..CAMERA_HEIGHT_RAW_VALID_MAX
        ) {
            return depth
        }
        val scale = CAMERA_HEIGHT_PRIOR_METERS / cameraHeightRaw
        if (!scale.isFinite() || scale !in 0.55f..1.45f) return depth
        return FloatArray(depth.size) { index ->
            val value = depth[index]
            if (value.isFinite() && value > 0f) value * scale else value
        }
    }

    private data class WallMeasurement(val width: Float, val height: Float, val depth: Float)

    private fun measureWall(
        depth: FloatArray,
        imageWidth: Int,
        imageHeight: Int,
        fx: Float,
        fy: Float,
    ): WallMeasurement {
        val margin = WALL_MARGIN.coerceIn(0f, 0.45f)
        val rectX = margin * imageWidth
        val rectY = margin * imageHeight
        val rectWidth = (1f - 2f * margin) * imageWidth
        val rectHeight = (1f - 2f * margin) * imageHeight

        val centerX = (imageWidth - 1) * 0.5f
        val centerY = (imageHeight - 1) * 0.5f
        val leftX = rectX.roundToInt().coerceIn(0, imageWidth - 1)
        val rightX = (rectX + rectWidth - 1f).roundToInt().coerceIn(0, imageWidth - 1)
        val topY = rectY.roundToInt().coerceIn(0, imageHeight - 1)
        val bottomY = (rectY + rectHeight - 1f).roundToInt().coerceIn(0, imageHeight - 1)
        val sampleCenterX = (rectX + rectWidth * 0.5f).roundToInt().coerceIn(0, imageWidth - 1)
        val sampleCenterY = (rectY + rectHeight * 0.5f).roundToInt().coerceIn(0, imageHeight - 1)

        val centerDepth = medianAt(depth, imageWidth, imageHeight, sampleCenterX, sampleCenterY)
            ?: centerDepthFallback(depth)
        val leftPlane = (leftX - centerX) * centerDepth / fx
        val rightPlane = (rightX - centerX) * centerDepth / fx
        val topPlane = (topY - centerY) * centerDepth / fy
        val bottomPlane = (bottomY - centerY) * centerDepth / fy
        return WallMeasurement(
            width = abs(rightPlane - leftPlane),
            height = abs(bottomPlane - topPlane),
            depth = centerDepth,
        )
    }

    private fun measureDepthSpread(
        depth: FloatArray,
        imageWidth: Int,
        imageHeight: Int,
    ): Float? {
        val margin = WALL_MARGIN.coerceIn(0f, 0.45f)
        val leftX = (margin * imageWidth).roundToInt().coerceIn(0, imageWidth - 1)
        val rightX = ((1f - margin) * imageWidth).roundToInt().coerceIn(0, imageWidth - 1) - 1
        val topY = (margin * imageHeight).roundToInt().coerceIn(0, imageHeight - 1)
        val bottomY = ((1f - margin) * imageHeight).roundToInt().coerceIn(0, imageHeight - 1) - 1
        if (leftX >= rightX || topY >= bottomY) return null

        val samples = ArrayList<Float>(512)
        var y = topY
        while (y <= bottomY) {
            var x = leftX
            while (x <= rightX) {
                val value = depth[y * imageWidth + x]
                if (value.isFinite() && value > 0f) samples += value
                x += 8
            }
            y += 8
        }
        if (samples.size < 16) return null
        samples.sort()
        return samples[((samples.size - 1) * 0.80f).roundToInt().coerceIn(0, samples.lastIndex)]
    }

    private fun sanitizeMeasurement(
        width: Float,
        height: Float,
        depth: Float,
        wallFallback: WallMeasurement,
    ): WallMeasurement {
        val spread = WallMeasurement(width, height, depth)
        if (isPlausible(spread)) return spread
        if (isPlausible(wallFallback)) return wallFallback
        return spread
    }

    private fun isPlausible(measurement: WallMeasurement): Boolean {
        return measurement.width in PLAUSIBLE_SPAN_MIN..PLAUSIBLE_SPAN_MAX &&
            measurement.height in PLAUSIBLE_SPAN_MIN..PLAUSIBLE_SPAN_MAX &&
            measurement.depth in PLAUSIBLE_SPAN_MIN..PLAUSIBLE_SPAN_MAX
    }

    private fun medianAt(
        depth: FloatArray,
        imageWidth: Int,
        imageHeight: Int,
        x: Int,
        y: Int,
        radius: Int = 5,
    ): Float? {
        val values = ArrayList<Float>((2 * radius + 1) * (2 * radius + 1))
        val yStart = max(0, y - radius)
        val yEnd = min(imageHeight - 1, y + radius)
        val xStart = max(0, x - radius)
        val xEnd = min(imageWidth - 1, x + radius)
        for (row in yStart..yEnd) {
            for (col in xStart..xEnd) {
                val value = depth[row * imageWidth + col]
                if (value.isFinite() && value > 0f) values += value
            }
        }
        if (values.isEmpty()) return null
        values.sort()
        return values[values.size / 2]
    }

    private fun centerDepthFallback(depth: FloatArray): Float {
        val valid = depth.filter { it.isFinite() && it > 0f }
        if (valid.isEmpty()) return 3.0f
        val sorted = valid.sorted()
        return sorted[sorted.size / 2]
    }

    private fun copyAssetToCache(context: Context, assetPath: String): File {
        val outFile = File(context.cacheDir, assetPath.replace('/', '_'))
        if (outFile.exists() && outFile.length() > 0L) return outFile
        outFile.parentFile?.mkdirs()
        context.assets.open(assetPath).use { input ->
            FileOutputStream(outFile).use { output -> input.copyTo(output) }
        }
        return outFile
    }
}
