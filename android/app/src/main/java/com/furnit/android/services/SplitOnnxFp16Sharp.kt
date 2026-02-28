package com.furnit.android.services

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import com.furnit.android.utils.LogUtil
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/**
 * FP16 split SHARP inference. Same 4-part pipeline as SplitOnnxSharp but with
 * FP16 quantized models (~1.3 GB total vs ~2.6 GB FP32).
 *
 * FP16 halves memory bandwidth (the main ARM CPU bottleneck) and allows more
 * aggressive session options for Part 4 since peak memory is roughly halved.
 *
 * No chunking needed -- FP16 models are small enough to fit without sub-splitting.
 */
class SplitOnnxFp16Sharp private constructor(private val context: Context) {

    companion object {
        private const val TAG = "SplitOnnxFp16Sharp"

        private const val NUM_PARTS = 4
        private val PART_FILENAMES = arrayOf(
            "sharp_part1_fp16.onnx" to "sharp_part1_fp16.onnx.data",
            "sharp_part2_fp16.onnx" to "sharp_part2_fp16.onnx.data",
            "sharp_part3_fp16.onnx" to "sharp_part3_fp16.onnx.data",
            "sharp_part4_fp16.onnx" to "sharp_part4_fp16.onnx.data"
        )

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
        private var instance: SplitOnnxFp16Sharp? = null

        fun getInstance(context: Context): SplitOnnxFp16Sharp {
            return instance ?: synchronized(this) {
                instance ?: SplitOnnxFp16Sharp(context.applicationContext).also {
                    instance = it
                    LogUtil.d(TAG, "SplitOnnxFp16Sharp singleton created")
                }
            }
        }
    }

    private val modelsDir: File by lazy {
        context.getExternalFilesDir("models") ?: File(context.filesDir, "models")
    }

    private val tempDir: File by lazy {
        File(context.cacheDir, "sharp_fp16_temp").also { it.mkdirs() }
    }

    private val ortEnv: OrtEnvironment by lazy {
        OrtEnvironment.getEnvironment()
    }

    @Volatile
    private var preloadedPart1Session: OrtSession? = null

    private val reusableSaveChunk: ByteBuffer by lazy {
        ByteBuffer.allocateDirect(4 * 1024 * 1024).apply { order(ByteOrder.LITTLE_ENDIAN) }
    }
    private val reusableTempArray = FloatArray(1024 * 1024)
    private val reusableHeaderBuffer: ByteBuffer by lazy {
        ByteBuffer.allocate(72).apply { order(ByteOrder.LITTLE_ENDIAN) }
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
        val ready = PART_FILENAMES.all { (model, data) ->
            val modelFile = File(modelsDir, model)
            val dataFile = File(modelsDir, data)
            val modelExists = modelFile.exists()
            val dataExists = dataFile.exists()
            // FP16 models may inline weights (no .data) if small enough
            val partReady = modelExists && (dataExists || modelFile.length() > 50_000_000)
            LogUtil.d(TAG, "isModelReady: $model exists=$modelExists size=${if (modelExists) modelFile.length() else 0} " +
                  "$data exists=$dataExists => $partReady")
            partReady
        }
        LogUtil.d(TAG, "isModelReady: result=$ready")
        return ready
    }

    fun getMissingFiles(): List<String> {
        val missing = mutableListOf<String>()
        for ((model, data) in PART_FILENAMES) {
            if (!File(modelsDir, model).exists()) missing.add(model)
        }
        return missing
    }

    fun getModelsDirPath(): String = modelsDir.absolutePath

    /** FP16: EXTENDED_OPT (not ALL_OPT) to avoid com.microsoft fused ops that lack FP16 CPU kernels. */
    private fun buildSessionOptions(partNumber: Int = 0): OrtSession.SessionOptions = OrtSession.SessionOptions().apply {
        val intraThreads = if (partNumber == 4) 4 else Runtime.getRuntime().availableProcessors()
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.EXTENDED_OPT)
        setIntraOpNumThreads(intraThreads)
        setInterOpNumThreads(1)
        setCPUArenaAllocator(partNumber != 4)
        setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
        setMemoryPatternOptimization(true)
        try {
            addConfigEntry("session.use_mmap", "1")
            addConfigEntry("session.enable_mem_reuse", "1")
            addConfigEntry("session.intra_op.allow_spinning", "1")
        } catch (e: Exception) {
            LogUtil.w(TAG, "Could not set session config: ${e.message}")
        }
        LogUtil.d(TAG, "buildSessionOptions(part=$partNumber): threads=$intraThreads arena=${partNumber != 4}")
    }

    suspend fun preloadSessions(progress: ((String) -> Unit)? = null) = withContext(Dispatchers.IO) {
        if (!isModelReady()) {
            LogUtil.w(TAG, "preloadSessions: model not ready, skipping")
            return@withContext
        }
        if (preloadedPart1Session != null) {
            LogUtil.d(TAG, "preloadSessions: Part 1 already loaded, skipping")
            return@withContext
        }
        val t0 = System.currentTimeMillis()
        try {
            val modelFile = PART_FILENAMES[0].first
            progress?.invoke("Loading FP16 Part 1...")
            val modelPath = File(modelsDir, modelFile).absolutePath
            LogUtil.d(TAG, "preloadSessions: Part 1 from $modelPath")
            preloadedPart1Session = ortEnv.createSession(modelPath, buildSessionOptions(partNumber = 1))
            LogUtil.d(TAG, "preloadSessions: Part 1 loaded in ${System.currentTimeMillis() - t0}ms. Memory: ${getMemoryInfo()}")
            progress?.invoke("FP16 Part 1 ready")
        } catch (e: Exception) {
            LogUtil.e(TAG, "preloadSessions failed: ${e.message}", e)
            try { preloadedPart1Session?.close() } catch (_: Exception) {}
            preloadedPart1Session = null
        }
    }

    fun releaseSessions() {
        try { preloadedPart1Session?.close() } catch (_: Exception) {}
        preloadedPart1Session = null
        LogUtil.d(TAG, "releaseSessions: Part 1 session closed")
    }

    private fun getMemoryInfo(): String {
        val runtime = Runtime.getRuntime()
        val usedMB = (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024
        val maxMB = runtime.maxMemory() / 1024 / 1024
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val memInfo = android.app.ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        return "JVM: ${usedMB}/${maxMB}MB, System: ${memInfo.availMem / 1024 / 1024}MB available"
    }

    suspend fun inferStreaming(
        bitmap: Bitmap,
        progressCallback: ((Float, String) -> Unit)? = null,
        isCancelled: () -> Boolean = { false }
    ): StreamingResult? = withContext(Dispatchers.IO) {
        if (!isModelReady()) {
            LogUtil.e(TAG, "FP16 model parts not ready. Missing: ${getMissingFiles()}")
            progressCallback?.invoke(0f, "FP16 model not ready")
            return@withContext null
        }

        val startTime = System.currentTimeMillis()
        try {
            LogUtil.d(TAG, "inferStreaming ENTER (FP16). Memory: ${getMemoryInfo()}")
            progressCallback?.invoke(0.05f, "Initializing (FP16)...")

            tempDir.listFiles()?.forEach { it.delete() }

            progressCallback?.invoke(0.1f, "Preprocessing image...")
            val scaledBitmap = SharpImagePreprocessor.resizeForSharp(bitmap)
            val inputBuffer = preprocessImageToBuffer(scaledBitmap)
            scaledBitmap.recycle()

            val inputTensorFile = File(tempDir, "input_image.tensor")
            saveFloatBufferToFile(inputBuffer, inputTensorFile, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))

            var currentInputs: Map<String, File> = mapOf("image" to inputTensorFile)
            val partTimes = LongArray(NUM_PARTS)
            val imageFile = inputTensorFile

            for (partIdx in 0 until NUM_PARTS) {
                if (isCancelled()) {
                    LogUtil.d(TAG, "CANCELLED before part ${partIdx + 1}")
                    return@withContext null
                }
                val partStartTime = System.currentTimeMillis()
                val partNumber = partIdx + 1
                val (modelFile, _) = PART_FILENAMES[partIdx]
                val modelPath = File(modelsDir, modelFile).absolutePath
                val baseProgress = 0.1f + (partIdx * 0.2f)

                progressCallback?.invoke(baseProgress, "FP16 Part $partNumber/4...")
                LogUtil.d(TAG, "=== FP16 Part $partNumber ($modelFile) === Memory: ${getMemoryInfo()}")

                val outputs = runModelPart(modelPath, currentInputs, partNumber, progressCallback, baseProgress)

                if (outputs == null) {
                    LogUtil.e(TAG, "FP16 Part $partNumber FAILED")
                    return@withContext null
                }

                if (partNumber == 1) {
                    preloadedPart1Session?.close()
                    preloadedPart1Session = null
                }

                currentInputs = currentInputs + outputs + ("image" to imageFile)
                partTimes[partIdx] = System.currentTimeMillis() - partStartTime
                LogUtil.d(TAG, "FP16 Part $partNumber done: ${partTimes[partIdx]}ms, tensors: ${currentInputs.size}, ${getMemoryInfo()}")
            }

            if (isCancelled()) return@withContext null

            progressCallback?.invoke(0.85f, "Processing Gaussian output...")
            val plyStart = System.currentTimeMillis()
            val result = processGaussianOutput(currentInputs, progressCallback)
            val plyTime = System.currentTimeMillis() - plyStart

            tempDir.listFiles()?.forEach { it.delete() }

            val elapsed = System.currentTimeMillis() - startTime
            LogUtil.d(TAG, "FP16 inference completed in ${elapsed}ms")
            LogUtil.d(TAG, "  Breakdown: P1=${partTimes[0]}ms P2=${partTimes[1]}ms P3=${partTimes[2]}ms P4=${partTimes[3]}ms PLY=${plyTime}ms")
            progressCallback?.invoke(1.0f, "Done!")

            return@withContext result

        } catch (e: OutOfMemoryError) {
            LogUtil.e(TAG, "OUT OF MEMORY during FP16 inference", e)
            return@withContext null
        } catch (e: Exception) {
            LogUtil.e(TAG, "FP16 inference failed: ${e.message}", e)
            return@withContext null
        }
    }

    private fun runModelPart(
        modelPath: String,
        inputs: Map<String, File>,
        partNumber: Int,
        progressCallback: ((Float, String) -> Unit)?,
        baseProgress: Float
    ): Map<String, File>? {
        LogUtil.d(TAG, "runModelPart ENTER part=$partNumber modelPath=$modelPath")
        var session: OrtSession? = if (partNumber == 1) preloadedPart1Session else null
        val usePreloaded = session != null
        if (!usePreloaded) {
            val sessionCreateStart = System.currentTimeMillis()
            session = ortEnv.createSession(modelPath, buildSessionOptions(partNumber))
            val sessionCreateTime = System.currentTimeMillis() - sessionCreateStart
            LogUtil.d(TAG, "Session created for part $partNumber in ${sessionCreateTime}ms")
        }

        try {
            val activeSession = session!!
            val inputNames = activeSession.inputNames.toList()
            val outputNames = activeSession.outputNames.toList()
            LogUtil.d(TAG, "Part $partNumber - Inputs: ${inputNames.size}, Outputs: ${outputNames.size}")

            val inputTensors = mutableMapOf<String, OnnxTensor>()
            for (name in inputNames) {
                val file = inputs[name]
                if (file == null || !file.exists()) {
                    LogUtil.e(TAG, "MISSING input tensor: $name")
                    return null
                }
                val tensor = loadTensorFromFile(ortEnv, file, name) ?: return null
                inputTensors[name] = tensor
            }

            if (partNumber == 4) {
                System.gc()
                System.runFinalization()
                Thread.sleep(150)
                LogUtil.d(TAG, "Part 4 pre-forward memory: ${getMemoryInfo()}")
            }

            LogUtil.d(TAG, "Part $partNumber: calling session.run with ${inputTensors.size} inputs")
            val inferStart = System.currentTimeMillis()
            val outputs = activeSession.run(inputTensors)
            val inferTime = System.currentTimeMillis() - inferStart
            LogUtil.d(TAG, "Part $partNumber inference: ${inferTime}ms")

            val outputFiles = mutableMapOf<String, File>()
            for (outputName in outputNames) {
                val tensor = outputs[outputName].get() as? OnnxTensor
                if (tensor != null) {
                    val file = File(tempDir, "part${partNumber}_${outputName.replace("/", "_")}.tensor")
                    saveTensorToFile(tensor, file)
                    outputFiles[outputName] = file
                }
            }

            inputTensors.values.forEach { it.close() }
            inputTensors.clear()
            outputs.close()

            return outputFiles

        } catch (e: Exception) {
            LogUtil.e(TAG, "runModelPart part=$partNumber EXCEPTION: ${e.message}", e)
            return null
        } finally {
            if (!usePreloaded) {
                session?.close()
            }
        }
    }

    private fun preprocessImageToBuffer(bitmap: Bitmap): FloatBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val pixelCount = width * height
        val pixels = IntArray(pixelCount)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val channelFloats = FloatArray(3 * pixelCount)
        val inv255 = 1f / 255f
        for (i in 0 until pixelCount) {
            val pixel = pixels[i]
            channelFloats[i] = ((pixel shr 16) and 0xFF) * inv255
            channelFloats[pixelCount + i] = ((pixel shr 8) and 0xFF) * inv255
            channelFloats[2 * pixelCount + i] = (pixel and 0xFF) * inv255
        }

        return FloatBuffer.wrap(channelFloats)
    }

    private fun saveFloatBufferToFile(buffer: FloatBuffer, file: File, shape: LongArray) {
        buffer.rewind()
        RandomAccessFile(file, "rw").use { raf ->
            val channel = raf.channel
            reusableHeaderBuffer.clear()
            reusableHeaderBuffer.putInt(shape.size)
            shape.forEach { reusableHeaderBuffer.putLong(it) }
            reusableHeaderBuffer.flip()
            channel.write(reusableHeaderBuffer)

            val totalFloats = buffer.remaining()
            val chunkSize = reusableTempArray.size
            var written = 0
            while (written < totalFloats) {
                val floatsToWrite = minOf(chunkSize, totalFloats - written)
                buffer.get(reusableTempArray, 0, floatsToWrite)
                reusableSaveChunk.clear()
                reusableSaveChunk.asFloatBuffer().put(reusableTempArray, 0, floatsToWrite)
                reusableSaveChunk.position(0)
                reusableSaveChunk.limit(floatsToWrite * 4)
                channel.write(reusableSaveChunk)
                written += floatsToWrite
            }
        }
    }

    private fun saveTensorToFile(tensor: OnnxTensor, file: File) {
        val shape = tensor.info.shape
        val floatBuffer = tensor.floatBuffer
        floatBuffer.rewind()
        RandomAccessFile(file, "rw").use { raf ->
            val channel = raf.channel
            reusableHeaderBuffer.clear()
            reusableHeaderBuffer.putInt(shape.size)
            shape.forEach { reusableHeaderBuffer.putLong(it) }
            reusableHeaderBuffer.flip()
            channel.write(reusableHeaderBuffer)

            val totalFloats = floatBuffer.remaining()
            val chunkSize = reusableTempArray.size
            var written = 0
            while (written < totalFloats) {
                val floatsToWrite = minOf(chunkSize, totalFloats - written)
                floatBuffer.get(reusableTempArray, 0, floatsToWrite)
                reusableSaveChunk.clear()
                reusableSaveChunk.asFloatBuffer().put(reusableTempArray, 0, floatsToWrite)
                reusableSaveChunk.position(0)
                reusableSaveChunk.limit(floatsToWrite * 4)
                channel.write(reusableSaveChunk)
                written += floatsToWrite
            }
        }
    }

    private fun loadTensorFromFile(env: OrtEnvironment, file: File, name: String): OnnxTensor? {
        try {
            RandomAccessFile(file, "r").use { raf ->
                val channel = raf.channel
                reusableHeaderBuffer.clear()
                reusableHeaderBuffer.limit(4)
                channel.read(reusableHeaderBuffer)
                reusableHeaderBuffer.flip()
                val numDims = reusableHeaderBuffer.int

                reusableHeaderBuffer.clear()
                reusableHeaderBuffer.limit(numDims * 8)
                channel.read(reusableHeaderBuffer)
                reusableHeaderBuffer.flip()
                val shape = LongArray(numDims) { reusableHeaderBuffer.long }

                val dataOffset = 4L + numDims * 8L
                val dataSize = shape.fold(1L) { acc, dim -> acc * dim } * 4
                val mappedBuffer = channel.map(FileChannel.MapMode.READ_ONLY, dataOffset, dataSize)
                mappedBuffer.order(ByteOrder.LITTLE_ENDIAN)
                return OnnxTensor.createTensor(env, mappedBuffer.asFloatBuffer(), shape)
            }
        } catch (e: Exception) {
            LogUtil.e(TAG, "loadTensorFromFile FAILED: $name: ${e.message}", e)
            return null
        }
    }

    private fun processGaussianOutput(
        outputs: Map<String, File>,
        progressCallback: ((Float, String) -> Unit)?
    ): StreamingResult? {
        val positionsFile = outputs.entries.find { it.key.contains("positions") }?.value
        val scalesFile = outputs.entries.find { it.key.contains("scales") }?.value
        val rotationsFile = outputs.entries.find { it.key.contains("rotations") }?.value
        val colorsFile = outputs.entries.find { it.key.contains("colors") }?.value
        val opacityFile = outputs.entries.find { it.key.contains("opacity") }?.value

        if (positionsFile == null || scalesFile == null || rotationsFile == null ||
            colorsFile == null || opacityFile == null) {
            LogUtil.e(TAG, "MISSING output tensors. Available: ${outputs.keys}")
            return null
        }

        val channels = mutableListOf<FileChannel>()
        try {
            val posChannel = RandomAccessFile(positionsFile, "r").channel.also { channels.add(it) }
            val scaleChannel = RandomAccessFile(scalesFile, "r").channel.also { channels.add(it) }
            val rotChannel = RandomAccessFile(rotationsFile, "r").channel.also { channels.add(it) }
            val colorChannel = RandomAccessFile(colorsFile, "r").channel.also { channels.add(it) }
            val opacityChannel = RandomAccessFile(opacityFile, "r").channel.also { channels.add(it) }

            val shapeBuffer = ByteBuffer.allocate(4 + 3 * 8).order(ByteOrder.LITTLE_ENDIAN)
            posChannel.read(shapeBuffer)
            shapeBuffer.flip()
            val numDims = shapeBuffer.int
            val shape = LongArray(numDims) { shapeBuffer.long }
            val gaussianCount = shape[1].toInt()

            LogUtil.d(TAG, "processGaussianOutput: $gaussianCount Gaussians")
            progressCallback?.invoke(0.9f, "Writing PLY ($gaussianCount Gaussians)...")

            val headerSize = 4L + numDims * 8L
            val posBuffer = posChannel.map(FileChannel.MapMode.READ_ONLY, headerSize, gaussianCount * 3L * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val scaleBuffer = scaleChannel.map(FileChannel.MapMode.READ_ONLY, headerSize, gaussianCount * 3L * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val rotBuffer = rotChannel.map(FileChannel.MapMode.READ_ONLY, headerSize, gaussianCount * 4L * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val colorBuffer = colorChannel.map(FileChannel.MapMode.READ_ONLY, headerSize, gaussianCount * 3L * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()

            val opacityShapeBuffer = ByteBuffer.allocate(4 + 3 * 8).order(ByteOrder.LITTLE_ENDIAN)
            opacityChannel.position(0)
            opacityChannel.read(opacityShapeBuffer)
            opacityShapeBuffer.flip()
            val opacityNumDims = opacityShapeBuffer.int
            val opacityHeaderSize = 4L + opacityNumDims * 8L
            val opacityBuffer = opacityChannel.map(FileChannel.MapMode.READ_ONLY, opacityHeaderSize, gaussianCount * 4L).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()

            val roomsDir = File(context.filesDir, "sharp_rooms")
            roomsDir.mkdirs()
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val roomFolder = File(roomsDir, "room_$timestamp")
            roomFolder.mkdirs()

            val plyFile = File(roomFolder, "room.ply")
            val classicPlyFile = File(roomFolder, "room_classic.ply")

            var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
            var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
            var minZ = Float.MAX_VALUE; var maxZ = -Float.MAX_VALUE

            val header = buildPlyHeader(gaussianCount)
            FileOutputStream(plyFile).use { fos ->
                val outChannel = fos.channel
                val headerBytes = header.toByteArray(Charsets.UTF_8)
                val headerBuffer = ByteBuffer.allocateDirect(headerBytes.size)
                headerBuffer.put(headerBytes)
                headerBuffer.flip()
                outChannel.write(headerBuffer)

                val batchBuffer = plyBatch
                batchBuffer.clear()
                val scaleBoost = 1.3f
                val minScale = 0.001f
                val lutScale = (LOGIT_LUT_SIZE - 1).toFloat()

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
                    while (batchBuffer.hasRemaining()) outChannel.write(batchBuffer)
                    batchBuffer.clear()
                    processed += currentBatch
                }
            }

            try {
                android.system.Os.link(plyFile.absolutePath, classicPlyFile.absolutePath)
            } catch (_: Exception) {
                plyFile.copyTo(classicPlyFile, overwrite = true)
            }

            LogUtil.d(TAG, "processGaussianOutput SUCCESS: $gaussianCount Gaussians, PLY=${plyFile.length() / 1024}KB")
            LogUtil.d(TAG, "Room bounds: ${maxX - minX}m x ${maxY - minY}m x ${maxZ - minZ}m")

            return StreamingResult(
                plyFile = plyFile,
                classicPlyFile = classicPlyFile,
                gaussianCount = gaussianCount,
                roomWidth = maxX - minX,
                roomHeight = maxY - minY,
                roomDepth = maxZ - minZ
            )

        } catch (e: Exception) {
            LogUtil.e(TAG, "processGaussianOutput FAILED", e)
            return null
        } finally {
            channels.forEach { try { it.close() } catch (_: Exception) {} }
        }
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
}
