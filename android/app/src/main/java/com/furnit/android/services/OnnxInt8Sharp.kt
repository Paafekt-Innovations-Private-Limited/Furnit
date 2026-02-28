package com.furnit.android.services

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.graphics.Bitmap
import com.furnit.android.utils.LogUtil
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/**
 * INT8-quantized SHARP inference using a single ONNX model.
 *
 * Single session loaded once, single forward pass producing 5 output tensors
 * (positions, scales, rotations, colors, opacity). Directly streams to PLY.
 *
 * Model: sharp_single_int8.onnx + sharp_single_int8.onnx.data (~600-700 MB total)
 */
class OnnxInt8Sharp private constructor(private val context: Context) {

    companion object {
        private const val TAG = "OnnxInt8Sharp"

        private const val MODEL_FILENAME = "sharp_single_int8.onnx"
        private const val WEIGHTS_FILENAME = "sharp_single_int8.onnx.data"

        private const val INPUT_SIZE = 1536
        private const val SH_C0 = 0.28209479177387814f
        private const val BYTES_PER_VERTEX = 62 * 4
        private const val PLY_BATCH_SIZE = 512
        private const val LOGIT_LUT_SIZE = 1024

        private val LOGIT_LUT = FloatArray(LOGIT_LUT_SIZE) { i ->
            val p = (i.toFloat() / (LOGIT_LUT_SIZE - 1)).coerceIn(1e-4f, 1f - 1e-4f)
            ln(p / (1f - p))
        }

        private const val LN_LUT_SIZE = 2048
        private const val LN_LUT_MIN = 0.001f
        private const val LN_LUT_MAX = 5.0f
        private val LN_LUT_SCALE = (LN_LUT_SIZE - 1).toFloat() / (LN_LUT_MAX - LN_LUT_MIN)
        private val LN_LUT = FloatArray(LN_LUT_SIZE) { i ->
            val x = LN_LUT_MIN + (LN_LUT_MAX - LN_LUT_MIN) * i / (LN_LUT_SIZE - 1)
            ln(x)
        }

        private fun lnLut(x: Float): Float {
            if (x <= LN_LUT_MIN) return LN_LUT[0]
            if (x >= LN_LUT_MAX) return LN_LUT[LN_LUT_SIZE - 1]
            return LN_LUT[((x - LN_LUT_MIN) * LN_LUT_SCALE).toInt()]
        }

        @Volatile
        private var instance: OnnxInt8Sharp? = null

        fun getInstance(context: Context): OnnxInt8Sharp {
            return instance ?: synchronized(this) {
                instance ?: OnnxInt8Sharp(context.applicationContext).also {
                    instance = it
                    LogUtil.d(TAG, "OnnxInt8Sharp singleton created")
                }
            }
        }
    }

    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    private val modelsDir: File by lazy {
        context.getExternalFilesDir("models") ?: File(context.filesDir, "models")
    }

    private val zeroSHBuffer: ByteBuffer by lazy {
        ByteBuffer.allocateDirect(45 * 4).apply { order(ByteOrder.LITTLE_ENDIAN) }
    }
    private val plyBatch: ByteBuffer by lazy {
        ByteBuffer.allocateDirect(BYTES_PER_VERTEX * PLY_BATCH_SIZE).apply { order(ByteOrder.LITTLE_ENDIAN) }
    }
    private val plyPositions = FloatArray(PLY_BATCH_SIZE * 3)
    private val plyScales = FloatArray(PLY_BATCH_SIZE * 3)
    private val plyRotations = FloatArray(PLY_BATCH_SIZE * 4)
    private val plyColors = FloatArray(PLY_BATCH_SIZE * 3)
    private val plyOpacity = FloatArray(PLY_BATCH_SIZE)

    data class StreamingResult(
        val plyFile: File,
        val classicPlyFile: File,
        val gaussianCount: Int,
        val roomWidth: Float,
        val roomHeight: Float,
        val roomDepth: Float
    )

    fun isModelReady(): Boolean {
        val modelFile = File(modelsDir, MODEL_FILENAME)
        val modelExists = modelFile.exists()
        val weightsFile = File(modelsDir, WEIGHTS_FILENAME)
        val weightsExists = weightsFile.exists()
        LogUtil.d(TAG, "isModelReady: $MODEL_FILENAME exists=$modelExists size=${if (modelExists) modelFile.length() else 0}, " +
              "$WEIGHTS_FILENAME exists=$weightsExists size=${if (weightsExists) weightsFile.length() else 0}")
        // INT8 quantization may inline all weights into the graph (no .data file needed)
        val ready = modelExists && (weightsExists || modelFile.length() > 100_000_000)
        LogUtil.d(TAG, "isModelReady: result=$ready")
        return ready
    }

    fun getModelsDirPath(): String = modelsDir.absolutePath

    private fun getMemoryInfo(): String {
        val runtime = Runtime.getRuntime()
        val usedMB = (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024
        val maxMB = runtime.maxMemory() / 1024 / 1024
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val memInfo = android.app.ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        return "JVM: ${usedMB}/${maxMB}MB, System: ${memInfo.availMem / 1024 / 1024}MB available"
    }

    /**
     * Load the single INT8 ONNX session. Called once during initialization.
     */
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        if (ortSession != null) return@withContext true

        if (!isModelReady()) {
            LogUtil.e(TAG, "Model not ready — call isModelReady() first")
            return@withContext false
        }

        try {
            val modelPath = File(modelsDir, MODEL_FILENAME).absolutePath
            LogUtil.d(TAG, "Loading INT8 ONNX model from: $modelPath")
            LogUtil.d(TAG, "Memory before load: ${getMemoryInfo()}")

            ortEnvironment = OrtEnvironment.getEnvironment()

            val sessionOptions = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setIntraOpNumThreads(4)
                setInterOpNumThreads(1)
                setCPUArenaAllocator(true)
                setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                setMemoryPatternOptimization(true)
                try {
                    addConfigEntry("session.use_mmap", "1")
                    addConfigEntry("session.use_env_allocators", "1")
                    addConfigEntry("session.inter_op.allow_spinning", "0")
                    addConfigEntry("session.enable_mem_reuse", "1")
                } catch (e: Exception) {
                    LogUtil.w(TAG, "Could not set session config: ${e.message}")
                }
            }

            var nnapiEnabled = false
            try {
                sessionOptions.addNnapi(EnumSet.of(
                    NNAPIFlags.USE_FP16,
                    NNAPIFlags.CPU_DISABLED
                ))
                nnapiEnabled = true
                LogUtil.d(TAG, "NNAPI execution provider added (USE_FP16, CPU_DISABLED)")
            } catch (e: Exception) {
                LogUtil.w(TAG, "NNAPI not available, using CPU only: ${e.message}")
            }

            System.gc()
            Thread.sleep(100)

            val loadStart = System.currentTimeMillis()
            ortSession = ortEnvironment?.createSession(modelPath, sessionOptions)
            val loadTime = System.currentTimeMillis() - loadStart

            LogUtil.d(TAG, "INT8 ONNX session created in ${loadTime}ms (optLevel=ALL, arena=true, memPattern=true, threads=4, nnapi=$nnapiEnabled). Memory after load: ${getMemoryInfo()}")
            return@withContext true

        } catch (e: OutOfMemoryError) {
            LogUtil.e(TAG, "OUT OF MEMORY loading INT8 model", e)
            System.gc()
            return@withContext false
        } catch (e: Exception) {
            LogUtil.e(TAG, "Failed to initialize INT8 session: ${e.javaClass.simpleName}", e)
            return@withContext false
        }
    }

    /**
     * Single forward pass through the INT8 model, stream results to PLY.
     */
    suspend fun inferStreaming(
        bitmap: Bitmap,
        progressCallback: ((Float, String) -> Unit)? = null,
        isCancelled: () -> Boolean = { false }
    ): StreamingResult? = withContext(Dispatchers.IO) {
        val session = ortSession
        val env = ortEnvironment

        if (session == null || env == null) {
            LogUtil.e(TAG, "Session not initialized — call initialize() first")
            return@withContext null
        }

        try {
            val startTime = System.currentTimeMillis()
            LogUtil.d(TAG, "inferStreaming ENTER (single INT8). Memory: ${getMemoryInfo()}")

            progressCallback?.invoke(0.1f, "Preprocessing image (INT8)...")

            val scaledBitmap = SharpImagePreprocessor.resizeForSharp(bitmap)
            val preprocessStart = System.currentTimeMillis()
            val inputTensor = preprocessImageFast(env, scaledBitmap)
            val preprocessTime = System.currentTimeMillis() - preprocessStart
            LogUtil.d(TAG, "Preprocessing: ${preprocessTime}ms (${scaledBitmap.width}x${scaledBitmap.height})")
            scaledBitmap.recycle()

            if (isCancelled()) {
                inputTensor.close()
                return@withContext null
            }

            progressCallback?.invoke(0.2f, "Running SHARP INT8 inference...")
            LogUtil.d(TAG, "Running single-model INT8 inference...")

            val inferStart = System.currentTimeMillis()
            val inputs = mapOf("image" to inputTensor)
            val outputs = session.run(inputs)
            val inferTime = System.currentTimeMillis() - inferStart
            LogUtil.d(TAG, "INT8 inference completed in ${inferTime}ms. Memory: ${getMemoryInfo()}")

            if (isCancelled()) {
                inputTensor.close()
                outputs.close()
                return@withContext null
            }

            progressCallback?.invoke(0.5f, "Processing Gaussian output...")

            val positionsTensor = outputs["positions"].get() as OnnxTensor
            val scalesTensor = outputs["scales"].get() as OnnxTensor
            val rotationsTensor = outputs["rotations"].get() as OnnxTensor
            val colorsTensor = outputs["colors"].get() as OnnxTensor
            val opacityTensor = outputs["opacity"].get() as OnnxTensor

            val posBuffer = positionsTensor.floatBuffer
            val scaleBuffer = scalesTensor.floatBuffer
            val rotBuffer = rotationsTensor.floatBuffer
            val colorBuffer = colorsTensor.floatBuffer
            val opacityBuffer = opacityTensor.floatBuffer

            val posShape = positionsTensor.info.shape  // [1, N, 3]
            val gaussianCount = posShape[1].toInt()

            LogUtil.d(TAG, "Streaming $gaussianCount Gaussians to PLY...")
            progressCallback?.invoke(0.6f, "Writing PLY ($gaussianCount Gaussians)...")

            val roomsDir = File(context.filesDir, "sharp_rooms")
            roomsDir.mkdirs()
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val roomFolder = File(roomsDir, "room_$timestamp")
            roomFolder.mkdirs()

            val plyFile = File(roomFolder, "room.ply")
            val classicPlyFile = File(roomFolder, "room_classic.ply")

            val header = buildPlyHeader(gaussianCount)

            var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
            var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
            var minZ = Float.MAX_VALUE; var maxZ = -Float.MAX_VALUE

            FileOutputStream(plyFile).use { fos ->
                val channel = fos.channel

                val headerBytes = header.toByteArray(Charsets.UTF_8)
                val headerBuffer = ByteBuffer.allocateDirect(headerBytes.size)
                headerBuffer.put(headerBytes)
                headerBuffer.flip()
                channel.write(headerBuffer)

                val batchBuffer = plyBatch
                batchBuffer.clear()
                val scaleBoost = 1.3f
                val minScale = 0.001f
                val lutScale = (LOGIT_LUT_SIZE - 1).toFloat()

                val progressEvery = max(1, gaussianCount / 10)
                var processed = 0

                while (processed < gaussianCount) {
                    val currentBatch = minOf(PLY_BATCH_SIZE, gaussianCount - processed)

                    posBuffer.position(processed * 3)
                    posBuffer.get(plyPositions, 0, currentBatch * 3)
                    scaleBuffer.position(processed * 3)
                    scaleBuffer.get(plyScales, 0, currentBatch * 3)
                    rotBuffer.position(processed * 4)
                    rotBuffer.get(plyRotations, 0, currentBatch * 4)
                    colorBuffer.position(processed * 3)
                    colorBuffer.get(plyColors, 0, currentBatch * 3)
                    opacityBuffer.position(processed)
                    opacityBuffer.get(plyOpacity, 0, currentBatch)

                    for (j in 0 until currentBatch) {
                        val idx3 = j * 3
                        val x = plyPositions[idx3]
                        val y = -plyPositions[idx3 + 1]
                        val z = -plyPositions[idx3 + 2]

                        if (x < minX) minX = x; if (x > maxX) maxX = x
                        if (y < minY) minY = y; if (y > maxY) maxY = y
                        if (z < minZ) minZ = z; if (z > maxZ) maxZ = z

                        batchBuffer.putFloat(x)
                        batchBuffer.putFloat(y)
                        batchBuffer.putFloat(z)

                        batchBuffer.putFloat(0f); batchBuffer.putFloat(0f); batchBuffer.putFloat(0f)

                        val r = plyColors[idx3].coerceIn(0f, 1f)
                        val g = plyColors[idx3 + 1].coerceIn(0f, 1f)
                        val b = plyColors[idx3 + 2].coerceIn(0f, 1f)
                        batchBuffer.putFloat((r - 0.5f) / SH_C0)
                        batchBuffer.putFloat((g - 0.5f) / SH_C0)
                        batchBuffer.putFloat((b - 0.5f) / SH_C0)

                        zeroSHBuffer.clear()
                        batchBuffer.put(zeroSHBuffer)

                        val rawOpacity = plyOpacity[j].coerceIn(0f, 1f)
                        batchBuffer.putFloat(LOGIT_LUT[(rawOpacity * lutScale).toInt().coerceIn(0, LOGIT_LUT_SIZE - 1)])

                        batchBuffer.putFloat(lnLut(max(plyScales[idx3] * scaleBoost, minScale)))
                        batchBuffer.putFloat(lnLut(max(plyScales[idx3 + 1] * scaleBoost, minScale)))
                        batchBuffer.putFloat(lnLut(max(plyScales[idx3 + 2] * scaleBoost, minScale)))

                        val idx4 = j * 4
                        val rw = plyRotations[idx4]
                        val rx = plyRotations[idx4 + 1]
                        val ry = plyRotations[idx4 + 2]
                        val rz = plyRotations[idx4 + 3]
                        val mag = sqrt(rw * rw + rx * rx + ry * ry + rz * rz)
                        val invMag = if (mag > 1e-8f) 1f / mag else 1f
                        batchBuffer.putFloat(rw * invMag)
                        batchBuffer.putFloat(rx * invMag)
                        batchBuffer.putFloat(ry * invMag)
                        batchBuffer.putFloat(rz * invMag)
                    }

                    batchBuffer.flip()
                    batchBuffer.limit(currentBatch * BYTES_PER_VERTEX)
                    while (batchBuffer.hasRemaining()) channel.write(batchBuffer)
                    batchBuffer.clear()

                    processed += currentBatch
                    if (processed % progressEvery == 0 || processed == gaussianCount) {
                        val progress = 0.6f + (processed.toFloat() / gaussianCount) * 0.3f
                        progressCallback?.invoke(progress, "Writing PLY ($processed/$gaussianCount)...")
                    }
                }
            }

            plyFile.copyTo(classicPlyFile, overwrite = true)

            inputTensor.close()
            outputs.close()
            System.gc()

            val totalElapsed = System.currentTimeMillis() - startTime
            LogUtil.d(TAG, "INT8 single-model completed: $gaussianCount Gaussians in ${totalElapsed}ms " +
                  "(inference=${inferTime}ms, PLY=${totalElapsed - inferTime - (inferStart - startTime)}ms)")
            LogUtil.d(TAG, "Room bounds: ${maxX - minX}m x ${maxY - minY}m x ${maxZ - minZ}m")

            progressCallback?.invoke(1.0f, "Done!")

            return@withContext StreamingResult(
                plyFile = plyFile,
                classicPlyFile = classicPlyFile,
                gaussianCount = gaussianCount,
                roomWidth = maxX - minX,
                roomHeight = maxY - minY,
                roomDepth = maxZ - minZ
            )

        } catch (e: Exception) {
            LogUtil.e(TAG, "INT8 streaming inference failed", e)
            return@withContext null
        }
    }

    private fun preprocessImageFast(env: OrtEnvironment, bitmap: Bitmap): OnnxTensor {
        val width = bitmap.width
        val height = bitmap.height
        val pixelCount = width * height

        val pixelByteBuffer = ByteBuffer.allocateDirect(pixelCount * 4)
        pixelByteBuffer.order(ByteOrder.nativeOrder())
        bitmap.copyPixelsToBuffer(pixelByteBuffer)
        pixelByteBuffer.rewind()

        val floatBuffer = ByteBuffer.allocateDirect(3 * pixelCount * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        val rOffset = 0
        val gOffset = pixelCount
        val bOffset = pixelCount * 2
        for (i in 0 until pixelCount) {
            val argb = pixelByteBuffer.getInt()
            floatBuffer.put(rOffset + i, ((argb shr 16) and 0xFF) / 255f)
            floatBuffer.put(gOffset + i, ((argb shr 8) and 0xFF) / 255f)
            floatBuffer.put(bOffset + i, (argb and 0xFF) / 255f)
        }
        floatBuffer.rewind()
        return OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 3, height.toLong(), width.toLong()))
    }

    private fun buildPlyHeader(gaussianCount: Int): String {
        return buildString {
            append("ply\n")
            append("format binary_little_endian 1.0\n")
            append("element vertex $gaussianCount\n")
            append("property float x\n")
            append("property float y\n")
            append("property float z\n")
            append("property float nx\n")
            append("property float ny\n")
            append("property float nz\n")
            for (i in 0 until 3) append("property float f_dc_$i\n")
            for (i in 0 until 45) append("property float f_rest_$i\n")
            append("property float opacity\n")
            append("property float scale_0\n")
            append("property float scale_1\n")
            append("property float scale_2\n")
            append("property float rot_0\n")
            append("property float rot_1\n")
            append("property float rot_2\n")
            append("property float rot_3\n")
            append("end_header\n")
        }
    }

    fun release() {
        ortSession?.close()
        ortEnvironment?.close()
        ortSession = null
        ortEnvironment = null
        LogUtil.d(TAG, "INT8 session released")
    }
}
