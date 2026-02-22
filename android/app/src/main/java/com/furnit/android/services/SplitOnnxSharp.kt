package com.furnit.android.services

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
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
 * Split SHARP model inference for memory-constrained Android devices.
 *
 * The FP32 model is split into 4 parts (~600MB each) that are loaded and executed
 * sequentially. Intermediate tensors are saved to disk between parts.
 *
 * Memory strategy:
 * - Part 1 is preloaded at startup (largest model, hides session-creation latency)
 * - Parts 2-4 are loaded on demand and closed after each run
 * - Intermediate tensors are memory-mapped from disk to avoid JVM heap pressure
 */
class SplitOnnxSharp private constructor(private val context: Context) {

    companion object {
        private const val TAG = "SplitOnnxSharp"

        // Model part configuration
        private const val NUM_PARTS = 4
        private const val USE_INT8_PART4 = false
        private const val PART4_INT8_FILENAME = "sharp_part4_int8.onnx"
        private const val USE_CHUNKED_PART1 = true
        private const val PART1A_FILENAME = "sharp_part1a.onnx"
        private const val PART1B_FILENAME = "sharp_part1b.onnx"
        private const val USE_CHUNKED_PART4 = true
        private const val PART4A_FILENAME = "sharp_part4a.onnx"
        private const val PART4A_DATA_FILENAME = "sharp_part4a.onnx.data"
        private const val PART4B_FILENAME = "sharp_part4b.onnx"
        private const val PART4B_DATA_FILENAME = "sharp_part4b.onnx.data"
        private const val NORM_OUTPUT_TENSOR = "/predictor/monodepth_model/encoder/image_encoder/norm/LayerNormalization_output_0"
        private val PART_FILENAMES = arrayOf(
            "sharp_part1.onnx" to "sharp_part1.onnx.data",
            "sharp_part2.onnx" to "sharp_part2.onnx.data",
            "sharp_part3.onnx" to "sharp_part3.onnx.data",
            "sharp_part4.onnx" to "sharp_part4.onnx.data"
        )

        // Must match model export: Part 1 expects [1,3,1536,1536]. Do not change without re-exporting ONNX parts.
        private const val INPUT_SIZE = 1536
        private const val SH_C0 = 0.28209479177387814f
        private const val BYTES_PER_VERTEX = 62 * 4
        private const val LOGIT_LUT_SIZE = 1024

        // Pre-computed logit LUT: maps [0, 1] opacity to ln(p/(1-p))
        private val LOGIT_LUT = FloatArray(LOGIT_LUT_SIZE) { i ->
            val p = (i.toFloat() / (LOGIT_LUT_SIZE - 1)).coerceIn(1e-4f, 1f - 1e-4f)
            ln(p / (1f - p))
        }

        // Pre-computed natural log LUT for scale values: avoids ln() per vertex
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
        private var instance: SplitOnnxSharp? = null

        fun getInstance(context: Context): SplitOnnxSharp {
            return instance ?: synchronized(this) {
                instance ?: SplitOnnxSharp(context.applicationContext).also {
                    instance = it
                    Log.d(TAG, "SplitOnnxSharp singleton created")
                }
            }
        }
    }

    // Model directory
    private val modelsDir: File by lazy {
        context.getExternalFilesDir("models") ?: File(context.filesDir, "models")
    }

    // Temporary directory for intermediate tensors
    private val tempDir: File by lazy {
        File(context.cacheDir, "sharp_temp").also { it.mkdirs() }
    }

    // Reuse OrtEnvironment across all parts (singleton pattern)
    private val ortEnv: OrtEnvironment by lazy {
        OrtEnvironment.getEnvironment()
    }

    // Only Part 1 is preloaded (946MB). Loading all 4 (~2.6GB) exhausts native memory before Part 1 can allocate activations.
    @Volatile
    private var preloadedPart1Session: OrtSession? = null

    // Reusable 4MB DirectByteBuffer + 1M FloatArray for tensor saves.
    // Avoids allocating transient DirectByteBuffers on each of the many save calls.
    // DirectByteBuffers freed by GC finalizers can linger and cause native memory pressure.
    private val reusableSaveChunk: ByteBuffer by lazy {
        ByteBuffer.allocateDirect(4 * 1024 * 1024).apply { order(ByteOrder.LITTLE_ENDIAN) }
    }
    private val reusableTempArray = FloatArray(1024 * 1024)

    // Reusable header buffer for tensor load (shape metadata: 4 + max 8*8 = 68 bytes)
    private val reusableHeaderBuffer: ByteBuffer by lazy {
        ByteBuffer.allocate(72).apply { order(ByteOrder.LITTLE_ENDIAN) }
    }

    // Pre-allocated zero SH block (180 bytes) for PLY writer — replaces 45 putFloat(0f) calls
    private val zeroSHBlock = ByteArray(45 * 4)

    data class StreamingResult(
        val plyFile: File,
        val classicPlyFile: File,
        val gaussianCount: Int,
        val roomWidth: Float,
        val roomHeight: Float,
        val roomDepth: Float
    )

    /**
     * Check if all model parts are ready.
     */
    fun isModelReady(): Boolean {
        val modelsDirPath = modelsDir.absolutePath
        Log.d(TAG, "isModelReady: modelsDir=$modelsDirPath")
        val ready = PART_FILENAMES.all { (model, data) ->
            val modelFile = File(modelsDir, model)
            val dataFile = File(modelsDir, data)
            val modelExists = modelFile.exists()
            val dataExists = dataFile.exists()
            // Part 1 may be exported without external .data (weights in .onnx); parts 2–4 need both
            val partReady = modelExists && dataExists
            Log.d(TAG, "isModelReady: $model exists=$modelExists size=${if (modelExists) modelFile.length() else 0} $data exists=$dataExists size=${if (dataExists) dataFile.length() else 0} => $partReady")
            partReady
        }
        Log.d(TAG, "isModelReady: result=$ready")
        return ready
    }

    /** Session options: ALL_OPT + arena. All cores for Parts 1-3; Part 4 capped at 2 threads (peak GEMM memory). */
    private fun buildSessionOptions(partNumber: Int = 0): OrtSession.SessionOptions = OrtSession.SessionOptions().apply {
        val intraThreads = if (partNumber == 4) 2 else Runtime.getRuntime().availableProcessors()
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
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
            Log.w(TAG, "Could not set session config: ${e.message}")
        }
        Log.d(TAG, "buildSessionOptions(part=$partNumber): threads=$intraThreads arena=${partNumber != 4}")
    }

    /**
     * Pre-create Part 1 (or Part 1a if chunked) session at startup.
     * Parts 2-4 are loaded on demand during inference and closed after use.
     */
    suspend fun preloadSessions(progress: ((String) -> Unit)? = null) = withContext(Dispatchers.IO) {
        if (!isModelReady()) {
            Log.w(TAG, "preloadSessions: model not ready, skipping")
            return@withContext
        }
        if (preloadedPart1Session != null) {
            Log.d(TAG, "preloadSessions: Part 1 already loaded, skipping")
            return@withContext
        }
        val t0 = System.currentTimeMillis()
        val useChunked = USE_CHUNKED_PART1 && File(modelsDir, PART1A_FILENAME).exists()
        val modelFile = if (useChunked) PART1A_FILENAME else PART_FILENAMES[0].first
        val label = if (useChunked) "Part 1a" else "Part 1"
        try {
            progress?.invoke("Loading $label...")
            val modelPath = File(modelsDir, modelFile).absolutePath
            Log.d(TAG, "preloadSessions: $label from $modelPath")
            preloadedPart1Session = ortEnv.createSession(modelPath, buildSessionOptions(partNumber = 1))
            Log.d(TAG, "preloadSessions: $label loaded in ${System.currentTimeMillis() - t0}ms. Memory: ${getMemoryInfo()}")
            progress?.invoke("$label ready")
        } catch (e: Exception) {
            Log.e(TAG, "preloadSessions failed: ${e.message}", e)
            try { preloadedPart1Session?.close() } catch (_: Exception) {}
            preloadedPart1Session = null
        }
    }

    /** Release preloaded Part 1 session. */
    fun releaseSessions() {
        try {
            preloadedPart1Session?.close()
        } catch (e: Exception) {
            Log.w(TAG, "releaseSessions: Part 1 close failed: ${e.message}")
        }
        preloadedPart1Session = null
        Log.d(TAG, "releaseSessions: Part 1 session closed")
    }

    /** Preload Part 1 session (replaces old single-part warmup). */
    suspend fun preloadAndWarmup(progress: ((String) -> Unit)? = null) = withContext(Dispatchers.IO) {
        preloadSessions(progress)
    }

    /**
     * Get list of missing model files for download.
     */
    fun getMissingFiles(): List<String> {
        val missing = mutableListOf<String>()
        for ((model, data) in PART_FILENAMES) {
            if (!File(modelsDir, model).exists()) missing.add(model)
            if (!File(modelsDir, data).exists()) missing.add(data)
        }
        return missing
    }

    /** Path on device where split ONNX files must be pushed (for error messages and logs). */
    fun getModelsDirPath(): String = modelsDir.absolutePath

    /**
     * Get memory info for logging.
     */
    private fun getMemoryInfo(): String {
        val runtime = Runtime.getRuntime()
        val usedMB = (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024
        val maxMB = runtime.maxMemory() / 1024 / 1024

        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val memInfo = android.app.ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        val availSysMB = memInfo.availMem / 1024 / 1024

        return "JVM: ${usedMB}/${maxMB}MB, System: ${availSysMB}MB available"
    }

    /**
     * Run inference using the split model approach.
     *
     * 1. Load Part 1, run, save intermediate tensors, unload
     * 2. Load Part 2, load intermediates, run, save, unload
     * 3. Load Part 3, load intermediates, run, save, unload
     * 4. Load Part 4, load intermediates, run, get final output
     */
    suspend fun inferStreaming(
        bitmap: Bitmap,
        progressCallback: ((Float, String) -> Unit)? = null,
        isCancelled: () -> Boolean = { false }
    ): StreamingResult? = withContext(Dispatchers.IO) {
        if (!isModelReady()) {
            Log.e(TAG, "Model parts not ready. Missing: ${getMissingFiles()}")
            progressCallback?.invoke(0f, "Model not ready")
            return@withContext null
        }

        val startTime = System.currentTimeMillis()
        try {
            Log.d(TAG, "inferStreaming ENTER. Memory: ${getMemoryInfo()}")
            progressCallback?.invoke(0.05f, "Initializing...")

            // Clean up any old temp files + optimized model cache (force re-optimization
            // only if needed; the optimized models are cached in a separate dir).
            tempDir.listFiles()?.forEach { it.delete() }

            // Preprocess input image to INPUT_SIZE (1536) to match model input shape.
            progressCallback?.invoke(0.1f, "Preprocessing image...")
            val preprocessStart = System.currentTimeMillis()
            val scaledBitmap = SharpImagePreprocessor.resizeForSharp(bitmap)
            val inputBuffer = preprocessImageToBuffer(scaledBitmap)
            scaledBitmap.recycle()
            Log.d(TAG, "Preprocess: ${System.currentTimeMillis() - preprocessStart}ms")

            // Save input tensor to file for Part 1
            val inputTensorFile = File(tempDir, "input_image.tensor")
            saveFloatBufferToFile(inputBuffer, inputTensorFile, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))
            Log.d(TAG, "Saved input_image.tensor size=${inputTensorFile.length()}")

            var currentInputs: Map<String, File> = mapOf("image" to inputTensorFile)
            val partTimes = LongArray(NUM_PARTS)
            val imageFile = File(tempDir, "input_image.tensor")

            val chunkedPart1Available = USE_CHUNKED_PART1 &&
                File(modelsDir, PART1A_FILENAME).exists() &&
                File(modelsDir, PART1B_FILENAME).exists()

            val chunkedPart4Available = USE_CHUNKED_PART4 &&
                File(modelsDir, PART4A_FILENAME).exists() &&
                File(modelsDir, PART4B_FILENAME).exists()

            if (chunkedPart1Available) {
                Log.d(TAG, "Chunked Part 1 models found (1a=${File(modelsDir, PART1A_FILENAME).length()/1_000_000}MB, " +
                    "1b=${File(modelsDir, PART1B_FILENAME).length()/1_000_000}MB)")
            }
            if (chunkedPart4Available) {
                Log.d(TAG, "Chunked Part 4 models found (4a=${File(modelsDir, PART4A_FILENAME).length()/1_000_000}MB, " +
                    "4b=${File(modelsDir, PART4B_FILENAME).length()/1_000_000}MB)")
            }

            // Part 1: either chunked (1a + 1b) or single model
            if (chunkedPart1Available) {
                val part1StartTime = System.currentTimeMillis()
                val part1Outputs = runChunkedPart1(currentInputs, progressCallback, isCancelled)
                if (part1Outputs == null) {
                    Log.e(TAG, "Chunked Part 1 FAILED")
                    return@withContext null
                }
                currentInputs = currentInputs + part1Outputs + ("image" to imageFile)
                partTimes[0] = System.currentTimeMillis() - part1StartTime
                Log.d(TAG, "Part 1 (chunked) done: ${partTimes[0]}ms, tensors: ${currentInputs.size}, ${getMemoryInfo()}")
            }

            // Parts loop: skip Part 1 if chunked, skip Part 4 if chunked
            val startPartIdx = if (chunkedPart1Available) 1 else 0
            val endPartIdx = if (chunkedPart4Available) 3 else NUM_PARTS

            for (partIdx in startPartIdx until endPartIdx) {
                if (isCancelled()) {
                    Log.d(TAG, "inferStreaming CANCELLED before part ${partIdx + 1}")
                    return@withContext null
                }
                val partStartTime = System.currentTimeMillis()
                val partBaseProgress = 0.1f + (partIdx * 0.2f)
                val (defaultModelFile, _) = PART_FILENAMES[partIdx]
                val partNumber = partIdx + 1
                val modelFile = if (partNumber == 4 && USE_INT8_PART4) {
                    val int8File = File(modelsDir, PART4_INT8_FILENAME)
                    if (int8File.exists()) {
                        Log.d(TAG, "Part 4 using INT8 quantized model (${int8File.length() / 1_000_000}MB)")
                        PART4_INT8_FILENAME
                    } else {
                        Log.d(TAG, "Part 4 INT8 model not found, falling back to FP32")
                        defaultModelFile
                    }
                } else {
                    defaultModelFile
                }
                progressCallback?.invoke(partBaseProgress, "Part $partNumber/4: Loading model...")

                Log.d(TAG, "=== Part $partNumber === Memory: ${getMemoryInfo()}")

                val modelPath = File(modelsDir, modelFile).absolutePath
                val outputs = runModelPart(modelPath, currentInputs, partNumber, progressCallback, partBaseProgress)

                if (outputs == null) {
                    Log.e(TAG, "Part $partNumber FAILED - runModelPart returned null")
                    return@withContext null
                }

                if (partNumber == 1) {
                    preloadedPart1Session?.close()
                    preloadedPart1Session = null
                    Log.d(TAG, "Closed Part 1 preloaded session to free memory for Parts 2-4")
                }

                currentInputs = currentInputs + outputs + ("image" to imageFile)

                partTimes[partIdx] = System.currentTimeMillis() - partStartTime
                Log.d(TAG, "Part $partNumber done: ${partTimes[partIdx]}ms, tensors: ${currentInputs.size}, ${getMemoryInfo()}")
            }

            // Part 4: either chunked (4a ViT + 4b decoder) or already run in loop above
            if (chunkedPart4Available) {
                val part4StartTime = System.currentTimeMillis()
                val part4Outputs = runChunkedPart4(currentInputs, progressCallback, isCancelled)
                if (part4Outputs == null) {
                    Log.e(TAG, "Chunked Part 4 FAILED")
                    return@withContext null
                }
                currentInputs = currentInputs + part4Outputs + ("image" to imageFile)
                partTimes[3] = System.currentTimeMillis() - part4StartTime
                Log.d(TAG, "Part 4 (chunked) done: ${partTimes[3]}ms, tensors: ${currentInputs.size}, ${getMemoryInfo()}")
            }

            if (isCancelled()) {
                Log.d(TAG, "inferStreaming CANCELLED before processGaussianOutput")
                return@withContext null
            }
            // Final part outputs are the Gaussian attributes
            progressCallback?.invoke(0.85f, "Processing Gaussian output...")
            Log.d(TAG, "Before processGaussianOutput. Memory: ${getMemoryInfo()}")
            Log.d(TAG, "Final currentInputs keys=${currentInputs.keys} files=${currentInputs.entries.joinToString { "${it.key}=${it.value.absolutePath}(${it.value.length()})" }}")
            val plyStart = System.currentTimeMillis()

            val result = processGaussianOutput(currentInputs, progressCallback)

            val plyTime = System.currentTimeMillis() - plyStart

            // Clean up temp files
            tempDir.listFiles()?.forEach { it.delete() }

            val elapsed = System.currentTimeMillis() - startTime
            Log.d(TAG, "Split inference completed in ${elapsed}ms")
            Log.d(TAG, "After PLY write. Memory: ${getMemoryInfo()}")
            val part1Label = if (chunkedPart1Available) "P1(chunked)" else "P1"
            val part4Label = if (chunkedPart4Available) "P4(chunked)" else "P4"
            Log.d(TAG, "  Breakdown: $part1Label=${partTimes[0]}ms P2=${partTimes[1]}ms P3=${partTimes[2]}ms $part4Label=${partTimes[3]}ms PLY=${plyTime}ms")
            progressCallback?.invoke(1.0f, "Done!")

            return@withContext result

        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OUT OF MEMORY during split inference after ${System.currentTimeMillis() - startTime}ms", e)
            progressCallback?.invoke(0f, "Out of memory")
            return@withContext null
        } catch (e: Exception) {
            Log.e(TAG, "Split inference failed after ${System.currentTimeMillis() - startTime}ms", e)
            progressCallback?.invoke(0f, "Error: ${e.message}")
            return@withContext null
        }
    }

    /**
     * Run a single model part.
     */
    private fun runModelPart(
        modelPath: String,
        inputs: Map<String, File>,
        partNumber: Int,
        progressCallback: ((Float, String) -> Unit)?,
        baseProgress: Float
    ): Map<String, File>? {
        Log.d(TAG, "runModelPart ENTER part=$partNumber modelPath=$modelPath inputKeys=${inputs.keys}")
        var session: OrtSession? = if (partNumber == 1) preloadedPart1Session else null
        val usePreloaded = session != null
        if (!usePreloaded) {
            val sessionCreateStart = System.currentTimeMillis()
            progressCallback?.invoke(baseProgress + 0.02f, "Part $partNumber/4: Creating session...")
            Log.d(TAG, "Creating session for part $partNumber...")
            session = ortEnv.createSession(modelPath, buildSessionOptions(partNumber))
            val sessionCreateTime = System.currentTimeMillis() - sessionCreateStart
            Log.d(TAG, "Session created for part $partNumber in ${sessionCreateTime}ms")
            progressCallback?.invoke(baseProgress + 0.05f, "Part $partNumber/4: Session ready (${sessionCreateTime/1000}s)")
        } else {
            Log.d(TAG, "Using preloaded session for part $partNumber")
            progressCallback?.invoke(baseProgress + 0.05f, "Part $partNumber/4: Session ready")
        }

        try {
            val activeSession = session!!
            val inputNames = activeSession.inputNames.toList()
            val outputNames = activeSession.outputNames.toList()
            Log.d(TAG, "Part $partNumber - Inputs: $inputNames, Outputs: $outputNames")

            // Load input tensors from files
            val loadStart = System.currentTimeMillis()
            progressCallback?.invoke(baseProgress + 0.07f, "Part $partNumber/4: Loading ${inputNames.size} inputs...")
            val inputTensors = mutableMapOf<String, OnnxTensor>()
            for ((idx, name) in inputNames.withIndex()) {
                val file = inputs[name]
                if (file == null || !file.exists()) {
                    Log.e(TAG, "runModelPart part=$partNumber: MISSING input tensor name='$name' (available: ${inputs.keys})")
                    return null
                }
                Log.d(TAG, "runModelPart part=$partNumber: loading input $name from ${file.absolutePath} exists=${file.exists()} size=${file.length()}")
                val tensor = loadTensorFromFile(ortEnv, file, name)
                if (tensor != null) {
                    inputTensors[name] = tensor
                    Log.d(TAG, "runModelPart part=$partNumber: loaded $name shape=${tensor.info.shape.contentToString()}")
                } else {
                    Log.e(TAG, "runModelPart part=$partNumber: FAILED to load tensor $name from ${file.absolutePath}")
                    return null
                }
                if (idx % 10 == 0) {
                    Log.d(TAG, "Loaded ${idx + 1}/${inputNames.size} inputs")
                }
            }
            val loadTime = System.currentTimeMillis() - loadStart
            Log.d(TAG, "All ${inputNames.size} inputs loaded for part $partNumber in ${loadTime}ms")

            // Run inference
            progressCallback?.invoke(baseProgress + 0.10f, "Part $partNumber/4: Running inference...")
            Log.d(TAG, "Running Part $partNumber inference with ${inputTensors.size} inputs...")
            val inferStartTime = System.currentTimeMillis()

            if (partNumber == 4) {
                System.gc()
                System.runFinalization()
                Thread.sleep(150)
                Log.d(TAG, "Part 4 pre-forward memory: ${getMemoryInfo()}")
                val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
                val memoryInfo = android.app.ActivityManager.MemoryInfo()
                activityManager.getMemoryInfo(memoryInfo)
                if (memoryInfo.availMem < 1024L * 1024 * 1024) {
                    Log.e(TAG, "ABORTING Part 4: only ${memoryInfo.availMem / 1024 / 1024}MB available, need ~1GB+")
                    return null
                }
            }

            Log.d(TAG, "runModelPart part=$partNumber: calling session.run with ${inputTensors.size} inputs")
            val outputs = activeSession.run(inputTensors)
            Log.d(TAG, "runModelPart part=$partNumber: session.run returned outputs: $outputNames")

            val inferTime = System.currentTimeMillis() - inferStartTime
            Log.d(TAG, "Part $partNumber inference completed in ${inferTime}ms")
            progressCallback?.invoke(baseProgress + 0.15f, "Part $partNumber/4: Inference done (${inferTime/1000}s)")

            // Save output tensors to files
            val saveStart = System.currentTimeMillis()
            val outputFiles = mutableMapOf<String, File>()
            for (outputName in outputNames) {
                val tensor = outputs[outputName].get() as? OnnxTensor
                if (tensor != null) {
                    val file = File(tempDir, "part${partNumber}_${outputName.replace("/", "_")}.tensor")
                    saveTensorToFile(tensor, file)
                    outputFiles[outputName] = file
                    val shapeStr = tensor.info.shape.contentToString()
                    Log.d(TAG, "Saved output: $outputName shape=$shapeStr -> ${file.name} (${file.length() / 1024}KB)")
                } else {
                    Log.e(TAG, "runModelPart part=$partNumber: output $outputName is null!")
                }
            }
            val saveTime = System.currentTimeMillis() - saveStart
            Log.d(TAG, "Part $partNumber outputs saved in ${saveTime}ms")

            // Clean up — close tensors immediately to release mmap handles and memory
            inputTensors.values.forEach { it.close() }
            inputTensors.clear()
            outputs.close()

            Log.d(TAG, "runModelPart part=$partNumber DONE outputFiles=${outputFiles.keys}")
            return outputFiles

        } catch (e: Exception) {
            Log.e(TAG, "runModelPart part=$partNumber EXCEPTION: ${e.message}", e)
            Log.e(TAG, "Stack trace: ${e.stackTraceToString()}")
            return null
        } finally {
            if (!usePreloaded) {
                session?.close()
            }
        }
    }

    /**
     * Run Part 1 as two sub-chunks: Part 1a (preprocess + blocks 0-11) then Part 1b (blocks 12-18).
     * Destroys Part 1a session before loading Part 1b to reduce peak memory from 946MB to max(611, 336) MB.
     */
    private fun runChunkedPart1(
        inputs: Map<String, File>,
        progressCallback: ((Float, String) -> Unit)?,
        isCancelled: () -> Boolean
    ): Map<String, File>? {
        Log.d(TAG, "=== Part 1 (chunked: 1a blocks 0-11 + 1b blocks 12-18) === Memory: ${getMemoryInfo()}")

        // --- Part 1a: Preprocess + blocks 0-11 (uses preloaded session) ---
        progressCallback?.invoke(0.10f, "Part 1a/1b: Blocks 0-11...")
        Log.d(TAG, "=== Part 1a (preprocess + blocks 0-11) === Memory: ${getMemoryInfo()}")

        val part1aModelPath = File(modelsDir, PART1A_FILENAME).absolutePath
        val part1aOutputs = runModelPart(part1aModelPath, inputs, 1, progressCallback, 0.10f)

        preloadedPart1Session?.close()
        preloadedPart1Session = null
        Log.d(TAG, "Closed Part 1a preloaded session")

        if (part1aOutputs == null) {
            Log.e(TAG, "Part 1a FAILED")
            return null
        }
        Log.d(TAG, "Part 1a done. Outputs: ${part1aOutputs.size} tensors. Memory: ${getMemoryInfo()}")

        if (isCancelled()) {
            Log.d(TAG, "inferStreaming CANCELLED between Part 1a and 1b")
            return null
        }

        // --- Part 1b: Blocks 12-18 ---
        progressCallback?.invoke(0.20f, "Part 1b/1b: Blocks 12-18...")
        Log.d(TAG, "=== Part 1b (blocks 12-18) === Memory: ${getMemoryInfo()}")

        val part1bInputs = inputs + part1aOutputs
        val part1bModelPath = File(modelsDir, PART1B_FILENAME).absolutePath
        val part1bOutputs = runModelPart(part1bModelPath, part1bInputs, 1, progressCallback, 0.20f)
        if (part1bOutputs == null) {
            Log.e(TAG, "Part 1b FAILED")
            return null
        }
        Log.d(TAG, "Part 1b done. Outputs: ${part1bOutputs.size} tensors. Memory: ${getMemoryInfo()}")

        return part1aOutputs + part1bOutputs
    }

    /**
     * Run Part 4 as two sub-chunks: Part 4a (ViT blocks) then Part 4b (decoder).
     * Destroys Part 4a session before loading Part 4b to reduce peak memory.
     */
    private fun runChunkedPart4(
        inputs: Map<String, File>,
        progressCallback: ((Float, String) -> Unit)?,
        isCancelled: () -> Boolean
    ): Map<String, File>? {
        Log.d(TAG, "=== Part 4 (chunked: 4a ViT + 4b decoder) === Memory: ${getMemoryInfo()}")

        // --- Part 4a: ViT blocks ---
        progressCallback?.invoke(0.70f, "Part 4a/4b: ViT blocks...")
        Log.d(TAG, "=== Part 4a (ViT) === Memory: ${getMemoryInfo()}")

        val part4aModelPath = File(modelsDir, PART4A_FILENAME).absolutePath
        val part4aOutputs = runModelPart(part4aModelPath, inputs, 4, progressCallback, 0.70f)
        if (part4aOutputs == null) {
            Log.e(TAG, "Part 4a FAILED")
            return null
        }
        Log.d(TAG, "Part 4a done. Norm output: ${part4aOutputs.keys}. Memory: ${getMemoryInfo()}")

        if (isCancelled()) {
            Log.d(TAG, "inferStreaming CANCELLED between Part 4a and 4b")
            return null
        }

        // --- Part 4b: Decoder ---
        progressCallback?.invoke(0.80f, "Part 4b/4b: Decoder...")
        Log.d(TAG, "=== Part 4b (decoder) === Memory: ${getMemoryInfo()}")

        val part4bInputs = inputs + part4aOutputs
        val part4bModelPath = File(modelsDir, PART4B_FILENAME).absolutePath
        val part4bOutputs = runModelPart(part4bModelPath, part4bInputs, 4, progressCallback, 0.80f)
        if (part4bOutputs == null) {
            Log.e(TAG, "Part 4b FAILED")
            return null
        }
        Log.d(TAG, "Part 4b done. Gaussian outputs: ${part4bOutputs.keys}. Memory: ${getMemoryInfo()}")

        return part4bOutputs
    }

    /**
     * Preprocess image to float buffer.
     *
     * Optimized: single pass over pixels extracts R, G, B simultaneously into CHW layout.
     * Old approach: 3 separate passes = 3 × 2.36M iterations = 7.08M total.
     * New approach: 1 pass = 2.36M iterations + 1 bulk FloatBuffer.put() memcpy.
     * Then bulk-copy into FloatBuffer — like iOS Accelerate vImage batch conversion.
     */
    private fun preprocessImageToBuffer(bitmap: Bitmap): FloatBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val pixelCount = width * height
        Log.d(TAG, "preprocessImageToBuffer: bitmap ${width}x${height} pixelCount=$pixelCount")
        val pixels = IntArray(pixelCount)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // Single-pass: extract R, G, B channels simultaneously into CHW layout
        val channelFloats = FloatArray(3 * pixelCount)
        val inv255 = 1f / 255f
        for (i in 0 until pixelCount) {
            val pixel = pixels[i]
            channelFloats[i] = ((pixel shr 16) and 0xFF) * inv255                  // R
            channelFloats[pixelCount + i] = ((pixel shr 8) and 0xFF) * inv255      // G
            channelFloats[2 * pixelCount + i] = (pixel and 0xFF) * inv255           // B
        }

        // Bulk write — single memcpy instead of 7M individual put() calls
        val floatBuffer = FloatBuffer.wrap(channelFloats)
        Log.d(TAG, "preprocessImageToBuffer: output size=${channelFloats.size} first6=${channelFloats.take(6)}")
        return floatBuffer
    }

    /**
     * Save a FloatBuffer to file with shape metadata using streaming.
     * Uses reusable chunk buffer to avoid transient DirectByteBuffer allocations.
     */
    private fun saveFloatBufferToFile(buffer: FloatBuffer, file: File, shape: LongArray) {
        buffer.rewind()
        Log.d(TAG, "saveFloatBufferToFile: file=${file.name} shape=${shape.contentToString()} buffer.remaining=${buffer.remaining()}")

        RandomAccessFile(file, "rw").use { raf ->
            val channel = raf.channel

            // Write shape metadata (reusable buffer)
            reusableHeaderBuffer.clear()
            reusableHeaderBuffer.putInt(shape.size)
            shape.forEach { reusableHeaderBuffer.putLong(it) }
            reusableHeaderBuffer.flip()
            channel.write(reusableHeaderBuffer)

            // Write float data in chunks using reusable DirectByteBuffer
            val totalFloats = buffer.remaining()
            val chunkSize = reusableTempArray.size  // 1M floats

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
        Log.d(TAG, "saveFloatBufferToFile: wrote ${file.length()} bytes to ${file.name}")
    }

    /**
     * Save an ONNX tensor to file using streaming.
     * Uses reusable chunk buffer to avoid transient DirectByteBuffer allocations.
     */
    private fun saveTensorToFile(tensor: OnnxTensor, file: File) {
        val shape = tensor.info.shape
        val floatBuffer = tensor.floatBuffer
        floatBuffer.rewind()

        RandomAccessFile(file, "rw").use { raf ->
            val channel = raf.channel

            // Write shape metadata (reusable buffer)
            reusableHeaderBuffer.clear()
            reusableHeaderBuffer.putInt(shape.size)
            shape.forEach { reusableHeaderBuffer.putLong(it) }
            reusableHeaderBuffer.flip()
            channel.write(reusableHeaderBuffer)

            // Write float data in chunks using reusable DirectByteBuffer
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

    /**
     * Load a tensor from file.
     */
    private fun loadTensorFromFile(env: OrtEnvironment, file: File, name: String): OnnxTensor? {
        Log.d(TAG, "loadTensorFromFile: name=$name file=${file.absolutePath} exists=${file.exists()}")
        try {
            RandomAccessFile(file, "r").use { raf ->
                val channel = raf.channel

                // Read shape metadata (reusable buffer — avoids allocation per tensor)
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

                // Memory map the data portion
                val dataOffset = 4L + numDims * 8L
                val dataSize = shape.fold(1L) { acc, dim -> acc * dim } * 4

                val mappedBuffer = channel.map(FileChannel.MapMode.READ_ONLY, dataOffset, dataSize)
                mappedBuffer.order(ByteOrder.LITTLE_ENDIAN)
                val floatBuffer = mappedBuffer.asFloatBuffer()

                Log.d(TAG, "Loaded tensor $name: shape=${shape.contentToString()}, size=${dataSize / 1024}KB")

                return OnnxTensor.createTensor(env, floatBuffer, shape)
            }
        } catch (e: Exception) {
            Log.e(TAG, "loadTensorFromFile FAILED name=$name file=${file.absolutePath}: ${e.message}", e)
            return null
        }
    }

    /**
     * Process final Gaussian output tensors into PLY file.
     */
    private fun processGaussianOutput(
        outputs: Map<String, File>,
        progressCallback: ((Float, String) -> Unit)?
    ): StreamingResult? {
        Log.d(TAG, "processGaussianOutput ENTER outputKeys=${outputs.keys}")

        val positionsFile = outputs.entries.find { it.key.contains("positions") }?.value
        val scalesFile = outputs.entries.find { it.key.contains("scales") }?.value
        val rotationsFile = outputs.entries.find { it.key.contains("rotations") }?.value
        val colorsFile = outputs.entries.find { it.key.contains("colors") }?.value
        val opacityFile = outputs.entries.find { it.key.contains("opacity") }?.value

        if (positionsFile == null || scalesFile == null || rotationsFile == null ||
            colorsFile == null || opacityFile == null) {
            Log.e(TAG, "processGaussianOutput: MISSING output tensors. Available: ${outputs.keys}")
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

            Log.d(TAG, "processGaussianOutput: $gaussianCount Gaussians")
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

                val batchSize = 512
                val batchBuffer = ByteBuffer.allocateDirect(BYTES_PER_VERTEX * batchSize).order(ByteOrder.LITTLE_ENDIAN)
                val scaleBoost = 1.3f
                val minScale = 0.001f
                val lutScale = (LOGIT_LUT_SIZE - 1).toFloat()

                val localPositions = FloatArray(batchSize * 3)
                val localScales = FloatArray(batchSize * 3)
                val localRotations = FloatArray(batchSize * 4)
                val localColors = FloatArray(batchSize * 3)
                val localOpacity = FloatArray(batchSize)

                var processed = 0
                while (processed < gaussianCount) {
                    val currentBatch = minOf(batchSize, gaussianCount - processed)

                    posBuffer.position(processed * 3)
                    posBuffer.get(localPositions, 0, currentBatch * 3)
                    scaleBuffer.position(processed * 3)
                    scaleBuffer.get(localScales, 0, currentBatch * 3)
                    rotBuffer.position(processed * 4)
                    rotBuffer.get(localRotations, 0, currentBatch * 4)
                    colorBuffer.position(processed * 3)
                    colorBuffer.get(localColors, 0, currentBatch * 3)
                    opacityBuffer.position(processed)
                    opacityBuffer.get(localOpacity, 0, currentBatch)

                    for (j in 0 until currentBatch) {
                        val idx3 = j * 3
                        val x = localPositions[idx3]
                        val y = -localPositions[idx3 + 1]
                        val z = -localPositions[idx3 + 2]

                        if (x < minX) minX = x; if (x > maxX) maxX = x
                        if (y < minY) minY = y; if (y > maxY) maxY = y
                        if (z < minZ) minZ = z; if (z > maxZ) maxZ = z

                        batchBuffer.putFloat(x)
                        batchBuffer.putFloat(y)
                        batchBuffer.putFloat(z)
                        batchBuffer.putFloat(0f); batchBuffer.putFloat(0f); batchBuffer.putFloat(0f)

                        val r = localColors[idx3].coerceIn(0f, 1f)
                        val g = localColors[idx3 + 1].coerceIn(0f, 1f)
                        val b = localColors[idx3 + 2].coerceIn(0f, 1f)
                        batchBuffer.putFloat((r - 0.5f) / SH_C0)
                        batchBuffer.putFloat((g - 0.5f) / SH_C0)
                        batchBuffer.putFloat((b - 0.5f) / SH_C0)
                        batchBuffer.put(zeroSHBlock)

                        val rawOpacity = localOpacity[j].coerceIn(0f, 1f)
                        batchBuffer.putFloat(LOGIT_LUT[(rawOpacity * lutScale).toInt().coerceIn(0, LOGIT_LUT_SIZE - 1)])

                        batchBuffer.putFloat(lnLut(max(localScales[idx3] * scaleBoost, minScale)))
                        batchBuffer.putFloat(lnLut(max(localScales[idx3 + 1] * scaleBoost, minScale)))
                        batchBuffer.putFloat(lnLut(max(localScales[idx3 + 2] * scaleBoost, minScale)))

                        val idx4 = j * 4
                        val rw = localRotations[idx4]
                        val rx = localRotations[idx4 + 1]
                        val ry = localRotations[idx4 + 2]
                        val rz = localRotations[idx4 + 3]
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

            // Hard-link instead of 100+MB file copy; fall back to copy if link fails
            try {
                android.system.Os.link(plyFile.absolutePath, classicPlyFile.absolutePath)
            } catch (_: Exception) {
                plyFile.copyTo(classicPlyFile, overwrite = true)
            }

            Log.d(TAG, "processGaussianOutput SUCCESS: $gaussianCount Gaussians, PLY=${plyFile.length() / 1024}KB")
            Log.d(TAG, "Room bounds: ${maxX - minX}m x ${maxY - minY}m x ${maxZ - minZ}m")

            return StreamingResult(
                plyFile = plyFile,
                classicPlyFile = classicPlyFile,
                gaussianCount = gaussianCount,
                roomWidth = maxX - minX,
                roomHeight = maxY - minY,
                roomDepth = maxZ - minZ
            )

        } catch (e: Exception) {
            Log.e(TAG, "processGaussianOutput FAILED", e)
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
