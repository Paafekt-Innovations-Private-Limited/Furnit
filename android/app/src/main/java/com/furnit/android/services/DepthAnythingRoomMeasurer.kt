package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import com.furnit.android.RoomDefaults
import com.furnit.android.roomreconstruction.DepthAnythingRoomMeasurementPipeline
import com.furnit.android.roomreconstruction.DepthMeshData
import com.furnit.android.roomreconstruction.RoomMeasurementConstants
import com.furnit.android.utils.LogUtil
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Depth Anything metric indoor depth + Swift-parity room measurement pipeline
 * (GeoCalib, RTMDet masking, RoomHeight/RoomExtent/depth-spread calibration).
 */
object DepthAnythingRoomMeasurer {
    private const val TAG = "DepthAnythingRoomMeasure"
    private const val MODEL_INPUT_SIZE = 518

    private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    data class Result(
        val width: Float,
        val height: Float,
        val depth: Float,
        val measured: Boolean,
        val source: String,
        val focalSource: String = "unknown",
        val depthMesh: DepthMeshData? = null,
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
        cameraMetadata: Map<String, Double>? = null,
        includeForegroundMask: Boolean = false,
    ): Result {
        val working = downsample(image, RoomMeasurementConstants.MAX_WORKING_IMAGE_DIMENSION)
        return try {
            val depth = runDepthInference(context, working) ?: return fallback("depth_inference_failed")
            val pipelineResult = DepthAnythingRoomMeasurementPipeline.measure(
                context = context,
                workingImage = working,
                rawDepth = depth,
                imageUri = imageUri,
                cameraMetadata = cameraMetadata,
                includeForegroundMask = includeForegroundMask,
            )
            Result(
                width = pipelineResult.width,
                height = pipelineResult.height,
                depth = pipelineResult.depth,
                measured = pipelineResult.measured,
                source = pipelineResult.source,
                focalSource = pipelineResult.focalSource,
                depthMesh = pipelineResult.depthMesh,
            )
        } catch (error: Exception) {
            LogUtil.e(TAG, "Room measurement failed", error)
            fallback("exception:${error.javaClass.simpleName}")
        } finally {
            if (working !== image) working.recycle()
        }
    }

    fun measureFromFile(
        context: Context,
        imageFile: File,
        imageUri: Uri? = null,
        cameraMetadata: Map<String, Double>? = null,
        includeForegroundMask: Boolean = false,
    ): Result {
        val bitmap = decodeOrientedBitmap(imageFile) ?: return fallback("decode_failed")
        return try {
            measure(context, bitmap, imageUri, cameraMetadata, includeForegroundMask)
        } finally {
            bitmap.recycle()
        }
    }

    internal fun decodeOrientedBitmap(imageFile: File): Bitmap? {
        val decoded = android.graphics.BitmapFactory.decodeFile(imageFile.absolutePath) ?: return null
        val oriented = applyExifOrientation(decoded, imageFile)
        if (oriented !== decoded) decoded.recycle()
        return oriented
    }

    private fun applyExifOrientation(bitmap: Bitmap, imageFile: File): Bitmap {
        val exif = runCatching { ExifInterface(imageFile) }.getOrNull() ?: return bitmap
        val rotationDegrees = exif.rotationDegrees
        val isFlipped = exif.isFlipped
        if (rotationDegrees == 0 && !isFlipped) return bitmap
        val transform = Matrix().apply {
            if (isFlipped) postScale(-1f, 1f)
            if (rotationDegrees != 0) postRotate(rotationDegrees.toFloat())
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, transform, true)
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
            val srcY = (
                (y.toFloat() + 0.5f) * sourceHeight.toFloat() / targetHeight.toFloat() - 0.5f
                ).coerceIn(0f, (sourceHeight - 1).toFloat())
            val y0 = srcY.toInt()
            val y1 = min(y0 + 1, sourceHeight - 1)
            val yFrac = srcY - y0
            for (x in 0 until targetWidth) {
                val srcX = (
                    (x.toFloat() + 0.5f) * sourceWidth.toFloat() / targetWidth.toFloat() - 0.5f
                    ).coerceIn(0f, (sourceWidth - 1).toFloat())
                val x0 = srcX.toInt()
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
