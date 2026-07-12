package com.furnit.android.roomreconstruction

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import com.furnit.android.services.RoomGenerationAssets
import com.furnit.android.utils.LogUtil
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.roundToInt

object GeoCalibCalibrationService {
    private const val TAG = "GeoCalibCalibration"
    private const val INPUT_SIDE = 320

    private val sessionLock = Any()
    @Volatile private var cachedSession: OrtSession? = null
    @Volatile private var cachedEnv: OrtEnvironment? = null
    @Volatile private var loadAttempted = false

    fun estimateCalibration(context: Context, image: Bitmap): GeoCalibCalibrationResult? {
        if (!assetExists(context, RoomGenerationAssets.GEOCALIB_PINHOLE_CNN_ONNX)) {
            return null
        }
        val session = loadSession(context) ?: return null
        val sourceWidth = image.width
        val sourceHeight = image.height
        val input = preprocess(image) ?: return null
        var tensor: OnnxTensor? = null
        return try {
            val env = cachedEnv ?: return null
            tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input.data), input.shape)
            val inputName = session.inputNames.iterator().next()
            val outputs = session.run(mapOf(inputName to tensor))
            outputs.use { result ->
                val outputMap = mutableMapOf<String, Any?>()
                session.outputNames.forEachIndexed { index, name ->
                    outputMap[name] = result[index].value
                }
                val fields = decodeFields(outputMap) ?: return null
                val letterbox = ImageLetterboxLayout.layout(sourceWidth, sourceHeight, INPUT_SIDE)
                val optimized = GeoCalibLMSolver.solve(
                    fields = fields,
                    contentMinX = letterbox.offsetX,
                    contentMinY = letterbox.offsetY,
                    contentMaxX = letterbox.offsetX + letterbox.contentWidth - 1,
                    contentMaxY = letterbox.offsetY + letterbox.contentHeight - 1,
                ) ?: return null
                val focalPx = optimized.focalPixels * letterbox.focalScaleToSource
                if (!focalPx.isFinite() || focalPx <= 1f) return null
                GeoCalibCalibrationResult(
                    focalLengthXPixels = focalPx,
                    focalLengthYPixels = focalPx,
                    rollRadians = optimized.rollRadians,
                    pitchRadians = optimized.pitchRadians,
                    finalCost = optimized.finalCost,
                    iterations = optimized.iterations,
                    sourceWidth = sourceWidth,
                    sourceHeight = sourceHeight,
                )
            }
        } catch (error: Exception) {
            LogUtil.e(TAG, "GeoCalib inference failed", error)
            null
        } finally {
            tensor?.close()
        }
    }

    private data class PreprocessResult(val data: FloatArray, val shape: LongArray)

    private fun preprocess(bitmap: Bitmap): PreprocessResult? {
        val letterbox = ImageLetterboxLayout.layout(bitmap.width, bitmap.height, INPUT_SIDE)
        val canvasBitmap = Bitmap.createBitmap(INPUT_SIDE, INPUT_SIDE, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(canvasBitmap)
        canvas.drawColor(Color.BLACK)
        val scaled = Bitmap.createScaledBitmap(bitmap, letterbox.contentWidth, letterbox.contentHeight, true)
        canvas.drawBitmap(scaled, letterbox.offsetX.toFloat(), letterbox.offsetY.toFloat(), null)
        if (scaled !== bitmap) scaled.recycle()

        val pixels = IntArray(INPUT_SIDE * INPUT_SIDE)
        canvasBitmap.getPixels(pixels, 0, INPUT_SIDE, 0, 0, INPUT_SIDE, INPUT_SIDE)
        canvasBitmap.recycle()

        val data = FloatArray(3 * INPUT_SIDE * INPUT_SIDE)
        var offset = 0
        for (channel in 0 until 3) {
            for (pixel in pixels) {
                val component = when (channel) {
                    0 -> (pixel shr 16) and 0xFF
                    1 -> (pixel shr 8) and 0xFF
                    else -> pixel and 0xFF
                }
                data[offset++] = component / 255.0f
            }
        }
        return PreprocessResult(data, longArrayOf(1, 3, INPUT_SIDE.toLong(), INPUT_SIDE.toLong()))
    }

    private fun decodeFields(outputs: Map<String, Any?>): GeoCalibLMSolver.Fields? {
        val up = outputs["up_field"] ?: return null
        val latitude = outputs["latitude_field"] ?: return null
        val upConf = outputs["up_confidence"]
        val latConf = outputs["latitude_confidence"]

        val height = INPUT_SIDE
        val width = INPUT_SIDE
        val planeSize = width * height

        val upChannels = extractNCHWChannels(up, width, height) ?: return null
        if (upChannels.size < 2) return null
        val latChannel = extractNCHWChannels(latitude, width, height)?.firstOrNull() ?: return null
        val upConfChannel = extractNCHWChannels(upConf, width, height)?.firstOrNull()
        val latConfChannel = extractNCHWChannels(latConf, width, height)?.firstOrNull()

        val upX = upChannels[0]
        val upY = upChannels[1]
        val lat = latChannel
        val upConfidence = upConfChannel ?: FloatArray(planeSize) { 1f }
        val latConfidence = latConfChannel ?: FloatArray(planeSize) { 1f }
        return GeoCalibLMSolver.Fields(width, height, upX, upY, lat, upConfidence, latConfidence)
    }

    /** Extract [C][H*W] planes from ONNX NCHW output (1,C,H,W) or NHW tensors. */
    private fun extractNCHWChannels(value: Any?, width: Int, height: Int): List<FloatArray>? {
        val planeSize = width * height
        val batch = unwrapBatch(value) ?: return null
        if (batch is FloatArray) {
            return if (batch.size == planeSize) listOf(batch) else null
        }
        if (batch !is Array<*>) return null

        val channels = mutableListOf<FloatArray>()
        for (entry in batch) {
            val plane = flattenPlane(entry, width, height) ?: return null
            channels += plane
        }
        return channels
    }

    private fun unwrapBatch(value: Any?): Any? {
        return when (value) {
            null -> null
            is Array<*> -> if (value.isNotEmpty() && value[0] is Array<*>) value[0] else value
            else -> value
        }
    }

    private fun flattenPlane(value: Any?, width: Int, height: Int): FloatArray? {
        val planeSize = width * height
        return when (value) {
            is FloatArray -> if (value.size == planeSize) value else null
            is Array<*> -> {
                if (value.isEmpty()) return null
                if (value[0] is Float) {
                    FloatArray(value.size) { index -> value[index] as Float }
                } else {
                    val out = FloatArray(planeSize)
                    var offset = 0
                    for (row in value) {
                        val rowArray = row as? Array<*> ?: return null
                        for (cell in rowArray) {
                            if (offset >= planeSize) return null
                            out[offset++] = cell as Float
                        }
                    }
                    if (offset == planeSize) out else null
                }
            }
            else -> null
        }
    }

    private fun flattenTensor(value: Any?): FloatArray? {
        return when (value) {
            is FloatArray -> value
            is Array<*> -> {
                val nested = value.flatMap { flattenTensor(it)?.toList() ?: return null }
                nested.toFloatArray()
            }
            else -> null
        }
    }

    private fun loadSession(context: Context): OrtSession? {
        cachedSession?.let { return it }
        synchronized(sessionLock) {
            cachedSession?.let { return it }
            if (loadAttempted) return null
            loadAttempted = true
            return try {
                val modelFile = File(context.cacheDir, RoomGenerationAssets.GEOCALIB_PINHOLE_CNN_ONNX)
                if (!modelFile.exists()) {
                    context.assets.open(RoomGenerationAssets.GEOCALIB_PINHOLE_CNN_ONNX).use { input ->
                        modelFile.parentFile?.mkdirs()
                        modelFile.outputStream().use { output -> input.copyTo(output) }
                    }
                }
                val env = OrtEnvironment.getEnvironment()
                val options = SessionOptions().apply {
                    setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT)
                    setExecutionMode(SessionOptions.ExecutionMode.SEQUENTIAL)
                    setIntraOpNumThreads(2)
                }
                val session = env.createSession(modelFile.absolutePath, options)
                cachedEnv = env
                cachedSession = session
                session
            } catch (error: Exception) {
                LogUtil.w(TAG, "GeoCalib ONNX unavailable: ${error.message}")
                null
            }
        }
    }

    private fun assetExists(context: Context, assetPath: String): Boolean {
        return try {
            context.assets.open(assetPath).close()
            true
        } catch (_: Exception) {
            false
        }
    }
}
