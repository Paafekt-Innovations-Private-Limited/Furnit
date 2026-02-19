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
 * This approach enables running a 2.5GB model on devices with 4-6GB RAM by:
 * 1. Loading only one part at a time (~600MB)
 * 2. Saving intermediate tensors to disk (memory-mapped for efficiency)
 * 3. Unloading each part before loading the next
 */
class SplitOnnxSharp private constructor(private val context: Context) {

    companion object {
        private const val TAG = "SplitOnnxSharp"

        // Model part configuration
        private const val NUM_PARTS = 4
        private val PART_FILENAMES = arrayOf(
            "sharp_part1.onnx" to "sharp_part1.onnx.data",
            "sharp_part2.onnx" to "sharp_part2.onnx.data",
            "sharp_part3.onnx" to "sharp_part3.onnx.data",
            "sharp_part4.onnx" to "sharp_part4.onnx.data"
        )

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
            Log.d(TAG, "isModelReady: $model exists=$modelExists size=${if (modelExists) modelFile.length() else 0} $data exists=$dataExists size=${if (dataExists) dataFile.length() else 0}")
            modelExists && dataExists
        }
        Log.d(TAG, "isModelReady: result=$ready")
        return ready
    }

    /**
     * Preload Part 1 and run a dummy forward to warm up the runtime.
     * Call when user opens the SHARP screen to reduce first-generation latency.
     */
    suspend fun preloadAndWarmup(progress: ((String) -> Unit)? = null) = withContext(Dispatchers.IO) {
        Log.d(TAG, "preloadAndWarmup ENTER")
        if (!isModelReady()) {
            Log.w(TAG, "preloadAndWarmup: model not ready, skipping")
            return@withContext
        }

        val (modelFile, _) = PART_FILENAMES[0]
        val modelPath = File(modelsDir, modelFile).absolutePath
        Log.d(TAG, "preloadAndWarmup: loading Part1 from $modelPath")
        progress?.invoke("Loading Part 1...")
        val t0 = System.currentTimeMillis()

        var session: OrtSession? = null
        var dummyTensor: OnnxTensor? = null
        try {
            val sessionOptions = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.EXTENDED_OPT)
                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
                setInterOpNumThreads(1)
                setCPUArenaAllocator(false)
                setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            }
            session = ortEnv.createSession(modelPath, sessionOptions)
            Log.d(TAG, "PRELOAD Part1 load ${System.currentTimeMillis() - t0}ms. Memory: ${getMemoryInfo()}")

            progress?.invoke("Warming up...")
            val dummyData = FloatArray(3 * INPUT_SIZE * INPUT_SIZE)
            val dummyBuffer = FloatBuffer.wrap(dummyData)
            dummyTensor = OnnxTensor.createTensor(ortEnv, dummyBuffer, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))
            val t1 = System.currentTimeMillis()
            session.run(mapOf("image" to dummyTensor))
            Log.d(TAG, "PRELOAD Part1 warmup ${System.currentTimeMillis() - t1}ms")
        } catch (e: Exception) {
            Log.w(TAG, "ONNX preload warmup failed: ${e.message}")
        } finally {
            dummyTensor?.close()
            session?.close()
            System.gc()
            Log.d(TAG, "preloadAndWarmup DONE total=${System.currentTimeMillis() - t0}ms")
            progress?.invoke("Preload done")
        }
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

            // Preprocess input image (shared with Native Pt)
            progressCallback?.invoke(0.1f, "Preprocessing image...")
            val preprocessStart = System.currentTimeMillis()
            val scaledBitmap = SharpImagePreprocessor.resizeForSharp(bitmap)
            val inputBuffer = preprocessImageToBuffer(scaledBitmap)
            scaledBitmap.recycle()
            Log.d(TAG, "Preprocess: ${System.currentTimeMillis() - preprocessStart}ms")

            // Save input tensor to file for Part 1
            val inputTensorFile = File(tempDir, "input_image.tensor")
            saveFloatBufferToFile(inputBuffer, inputTensorFile, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))
            Log.d(TAG, "Saved input_image.tensor shape=[1,3,$INPUT_SIZE,$INPUT_SIZE] file=${inputTensorFile.absolutePath} size=${inputTensorFile.length()}")
            val inputSample = FloatArray(6)
            inputBuffer.rewind()
            inputBuffer.get(inputSample)
            inputBuffer.rewind()
            Log.d(TAG, "Input buffer first6 floats: ${inputSample.toList()}")

            var currentInputs: Map<String, File> = mapOf("image" to inputTensorFile)
            val partTimes = LongArray(NUM_PARTS)

            // Run each part sequentially
            for (partIdx in 0 until NUM_PARTS) {
                if (isCancelled()) {
                    Log.d(TAG, "inferStreaming CANCELLED before part ${partIdx + 1}")
                    return@withContext null
                }
                val partStartTime = System.currentTimeMillis()
                // Better progress: each part gets 20% (total 80% for 4 parts)
                val partBaseProgress = 0.1f + (partIdx * 0.2f)
                val (modelFile, _) = PART_FILENAMES[partIdx]
                progressCallback?.invoke(partBaseProgress, "Part ${partIdx + 1}/4: Loading model...")

                Log.d(TAG, "=== Part ${partIdx + 1} ===")
                Log.d(TAG, "Memory before: ${getMemoryInfo()}")
                Log.d(TAG, "Model file: $modelFile")

                // Request GC to reclaim previous session's memory
                System.gc()

                val modelPath = File(modelsDir, modelFile).absolutePath
                Log.d(TAG, "Part ${partIdx + 1}: currentInputs keys=${currentInputs.keys} files=${currentInputs.map { "${it.key}=${it.value.name}(${it.value.length()})" }}")
                val outputs = runModelPart(modelPath, currentInputs, partIdx + 1, progressCallback, partBaseProgress)

                if (outputs == null) {
                    Log.e(TAG, "Part ${partIdx + 1} FAILED - runModelPart returned null")
                    return@withContext null
                }
                Log.d(TAG, "Part ${partIdx + 1}: outputs keys=${outputs.keys} files=${outputs.map { "${it.key}=${it.value.name}(${it.value.length()})" }}")

                // IMPORTANT: Accumulate all outputs across parts - later parts may need
                // outputs from earlier parts (e.g., Part 3 needs weight tensors from Part 1)
                val imageFile = File(tempDir, "input_image.tensor")
                currentInputs = currentInputs + outputs + ("image" to imageFile)
                Log.d(TAG, "Part ${partIdx + 1}: merged currentInputs keys=${currentInputs.keys}")

                partTimes[partIdx] = System.currentTimeMillis() - partStartTime
                Log.d(TAG, "Part ${partIdx + 1} total: ${partTimes[partIdx]}ms (tensors: ${currentInputs.size})")
                Log.d(TAG, "Memory after Part ${partIdx + 1}: ${getMemoryInfo()}")
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
            Log.d(TAG, "  Breakdown: P1=${partTimes[0]}ms P2=${partTimes[1]}ms P3=${partTimes[2]}ms P4=${partTimes[3]}ms PLY=${plyTime}ms")
            progressCallback?.invoke(1.0f, "Done!")

            return@withContext result

        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OUT OF MEMORY during split inference after ${System.currentTimeMillis() - startTime}ms", e)
            System.gc()
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
        var session: OrtSession? = null

        try {
            val sessionCreateStart = System.currentTimeMillis()

            // Session options tuned for split models with external data files.
            // IMPORTANT: ALL_OPT is NOT safe here — it tries to internalize external tensor
            // references which breaks split models. EXTENDED_OPT does operator fusion and
            // constant folding but avoids the aggressive transformations.
            // Similarly, optimized_model_filepath cannot be used with external data files
            // because the cached model loses the external data references.
            val sessionOptions = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.EXTENDED_OPT)

                val numCores = Runtime.getRuntime().availableProcessors()
                // MAX mode: always use all available CPU cores for ONNX inference.
                // This maximizes parallelism for large GEMM/attention ops at the cost of
                // higher power draw and potential thermal throttling on long runs.
                val intraThreads = numCores
                setIntraOpNumThreads(intraThreads)
                setInterOpNumThreads(1)
                Log.d(TAG, "CPU: MAX mode - using $intraThreads intra-op threads for part $partNumber (device has $numCores cores)")

                setMemoryPatternOptimization(true)
                // Arena allocator OFF for split models — Part 4 (60 inputs, ~600MB model)
                // uses too much peak memory with arena pooling on memory-constrained devices.
                setCPUArenaAllocator(false)
                setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)

                try {
                    addConfigEntry("session.use_mmap", "1")
                    addConfigEntry("session.enable_mem_reuse", "1")
                    // Worker threads busy-wait instead of sleeping between ops.
                    // Reduces latency for sequential CPU ops in transformer layers.
                    addConfigEntry("session.intra_op.allow_spinning", "1")
                } catch (e: Exception) {
                    Log.w(TAG, "Could not set session config: ${e.message}")
                }
            }

            progressCallback?.invoke(baseProgress + 0.02f, "Part $partNumber/4: Creating session...")
            Log.d(TAG, "Creating optimized CPU session for part $partNumber...")
            session = ortEnv.createSession(modelPath, sessionOptions)
            val sessionCreateTime = System.currentTimeMillis() - sessionCreateStart
            Log.d(TAG, "Session created for part $partNumber in ${sessionCreateTime}ms")
            progressCallback?.invoke(baseProgress + 0.05f, "Part $partNumber/4: Session ready (${sessionCreateTime/1000}s)")

            // Get input/output names
            val inputNames = session.inputNames.toList()
            val outputNames = session.outputNames.toList()
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

            Log.d(TAG, "runModelPart part=$partNumber: calling session.run with ${inputTensors.size} inputs")
            val outputs = session.run(inputTensors)
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

            // Aggressive cleanup (Native Pt style) to reduce OOM risk
            System.gc()

            Log.d(TAG, "runModelPart part=$partNumber DONE outputFiles=${outputFiles.keys}")
            return outputFiles

        } catch (e: Exception) {
            Log.e(TAG, "runModelPart part=$partNumber EXCEPTION: ${e.message}", e)
            Log.e(TAG, "Stack trace: ${e.stackTraceToString()}")
            return null
        } finally {
            session?.close()
            session = null
            // Note: ortEnv is reused across parts, don't close it

            // Force GC after closing session (Native Pt style)
            System.gc()
        }
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
        try {
            // Find the output files (named with part4_ prefix)
            val positionsFile = outputs.entries.find { it.key.contains("positions") }?.value
            val scalesFile = outputs.entries.find { it.key.contains("scales") }?.value
            val rotationsFile = outputs.entries.find { it.key.contains("rotations") }?.value
            val colorsFile = outputs.entries.find { it.key.contains("colors") }?.value
            val opacityFile = outputs.entries.find { it.key.contains("opacity") }?.value

            Log.d(TAG, "processGaussianOutput: positions=$positionsFile scales=$scalesFile rotations=$rotationsFile colors=$colorsFile opacity=$opacityFile")

            if (positionsFile == null || scalesFile == null || rotationsFile == null ||
                colorsFile == null || opacityFile == null) {
                Log.e(TAG, "processGaussianOutput: MISSING output tensors. Available: ${outputs.keys}")
                return null
            }

            // Memory-map the output files
            val posChannel = RandomAccessFile(positionsFile, "r").channel
            val scaleChannel = RandomAccessFile(scalesFile, "r").channel
            val rotChannel = RandomAccessFile(rotationsFile, "r").channel
            val colorChannel = RandomAccessFile(colorsFile, "r").channel
            val opacityChannel = RandomAccessFile(opacityFile, "r").channel

            // Read position shape to get Gaussian count
            val shapeBuffer = ByteBuffer.allocate(4 + 3 * 8)
            shapeBuffer.order(ByteOrder.LITTLE_ENDIAN)
            posChannel.read(shapeBuffer)
            shapeBuffer.flip()
            val numDims = shapeBuffer.int
            val shape = LongArray(numDims) { shapeBuffer.long }
            val gaussianCount = shape[1].toInt()

            Log.d(TAG, "processGaussianOutput: positions shape=$shape gaussianCount=$gaussianCount")
            Log.d(TAG, "Processing $gaussianCount Gaussians...")
            progressCallback?.invoke(0.9f, "Writing PLY ($gaussianCount Gaussians)...")

            // Map data buffers (skip header)
            val headerSize = 4L + numDims * 8L
            val posBuffer = posChannel.map(FileChannel.MapMode.READ_ONLY, headerSize, gaussianCount * 3L * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val scaleBuffer = scaleChannel.map(FileChannel.MapMode.READ_ONLY, headerSize, gaussianCount * 3L * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val rotBuffer = rotChannel.map(FileChannel.MapMode.READ_ONLY, headerSize, gaussianCount * 4L * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val colorBuffer = colorChannel.map(FileChannel.MapMode.READ_ONLY, headerSize, gaussianCount * 3L * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()

            // Opacity might be [1, N] or [1, N, 1]
            val opacityShapeBuffer = ByteBuffer.allocate(4 + 3 * 8)
            opacityShapeBuffer.order(ByteOrder.LITTLE_ENDIAN)
            opacityChannel.position(0)
            opacityChannel.read(opacityShapeBuffer)
            opacityShapeBuffer.flip()
            val opacityNumDims = opacityShapeBuffer.int
            val opacityHeaderSize = 4L + opacityNumDims * 8L
            val opacityBuffer = opacityChannel.map(FileChannel.MapMode.READ_ONLY, opacityHeaderSize, gaussianCount * 4L).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()

            Log.d(TAG, "processGaussianOutput: mapped buffers posLimit=${posBuffer.limit()} scaleLimit=${scaleBuffer.limit()} rotLimit=${rotBuffer.limit()} colorLimit=${colorBuffer.limit()} opacityLimit=${opacityBuffer.limit()}")
            val posSample = FloatArray(3)
            posBuffer.duplicate().apply { position(0); get(posSample) }
            Log.d(TAG, "processGaussianOutput: sample pos[0..2]=${posSample.toList()}")

            // Create output directory
            val roomsDir = File(context.filesDir, "sharp_rooms")
            roomsDir.mkdirs()

            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val roomFolder = File(roomsDir, "room_$timestamp")
            roomFolder.mkdirs()

            val plyFile = File(roomFolder, "room.ply")
            val classicPlyFile = File(roomFolder, "room_classic.ply")

            // Bounds tracking
            var minX = Float.MAX_VALUE
            var maxX = -Float.MAX_VALUE
            var minY = Float.MAX_VALUE
            var maxY = -Float.MAX_VALUE
            var minZ = Float.MAX_VALUE
            var maxZ = -Float.MAX_VALUE

            // Write PLY with DirectByteBuffer + FileChannel for zero-copy writes
            val header = buildPlyHeader(gaussianCount)
            FileOutputStream(plyFile).use { fos ->
                val channel = fos.channel

                // Write header
                val headerBytes = header.toByteArray(Charsets.UTF_8)
                val headerBuffer = ByteBuffer.allocateDirect(headerBytes.size)
                headerBuffer.put(headerBytes)
                headerBuffer.flip()
                channel.write(headerBuffer)

                // DirectByteBuffer for zero-copy writes (batch 512 vertices ~127KB)
                val batchSize = 512
                val batchBuffer = ByteBuffer.allocateDirect(BYTES_PER_VERTEX * batchSize)
                batchBuffer.order(ByteOrder.LITTLE_ENDIAN)
                val scaleBoost = 1.3f
                val minScale = 0.001f
                val lutScale = (LOGIT_LUT_SIZE - 1).toFloat()

                // Pre-allocate local arrays for vectorized bulk reads
                val localPositions = FloatArray(batchSize * 3)
                val localScales = FloatArray(batchSize * 3)
                val localRotations = FloatArray(batchSize * 4)
                val localColors = FloatArray(batchSize * 3)
                val localOpacity = FloatArray(batchSize)

                var processed = 0
                while (processed < gaussianCount) {
                    val currentBatch = minOf(batchSize, gaussianCount - processed)

                    // Vectorized bulk reads: one buffer call per attribute
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

                    // Process from local arrays (no per-element buffer/JNI overhead)
                    for (j in 0 until currentBatch) {
                        val idx3 = j * 3
                        val x = localPositions[idx3]
                        val y = -localPositions[idx3 + 1]
                        val z = -localPositions[idx3 + 2]

                        if (x < minX) minX = x
                        if (x > maxX) maxX = x
                        if (y < minY) minY = y
                        if (y > maxY) maxY = y
                        if (z < minZ) minZ = z
                        if (z > maxZ) maxZ = z

                        batchBuffer.putFloat(x)
                        batchBuffer.putFloat(y)
                        batchBuffer.putFloat(z)

                        // Normals
                        batchBuffer.putFloat(0f)
                        batchBuffer.putFloat(0f)
                        batchBuffer.putFloat(0f)

                        // Colors -> SH DC
                        val r = localColors[idx3].coerceIn(0f, 1f)
                        val g = localColors[idx3 + 1].coerceIn(0f, 1f)
                        val b = localColors[idx3 + 2].coerceIn(0f, 1f)
                        batchBuffer.putFloat((r - 0.5f) / SH_C0)
                        batchBuffer.putFloat((g - 0.5f) / SH_C0)
                        batchBuffer.putFloat((b - 0.5f) / SH_C0)

                        // Higher order SH (45 zeros) — bulk put instead of 45 individual putFloat(0f)
                        batchBuffer.put(zeroSHBlock)

                        // Opacity -> logit via LUT
                        val rawOpacity = localOpacity[j].coerceIn(0f, 1f)
                        val lutIndex = (rawOpacity * lutScale).toInt().coerceIn(0, LOGIT_LUT_SIZE - 1)
                        batchBuffer.putFloat(LOGIT_LUT[lutIndex])

                        // Scale -> log via LN LUT (avoids ln() per vertex)
                        batchBuffer.putFloat(lnLut(max(localScales[idx3] * scaleBoost, minScale)))
                        batchBuffer.putFloat(lnLut(max(localScales[idx3 + 1] * scaleBoost, minScale)))
                        batchBuffer.putFloat(lnLut(max(localScales[idx3 + 2] * scaleBoost, minScale)))

                        // Rotation -> normalize
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

                    // Flush batch to disk via zero-copy channel write
                    batchBuffer.flip()
                    batchBuffer.limit(currentBatch * BYTES_PER_VERTEX)
                    while (batchBuffer.hasRemaining()) {
                        channel.write(batchBuffer)
                    }
                    batchBuffer.clear()
                    processed += currentBatch
                }
            }

            // Copy to classic PLY
            plyFile.copyTo(classicPlyFile, overwrite = true)

            // Close channels
            posChannel.close()
            scaleChannel.close()
            rotChannel.close()
            colorChannel.close()
            opacityChannel.close()

            Log.d(TAG, "processGaussianOutput SUCCESS: PLY written $gaussianCount Gaussians to ${plyFile.absolutePath} size=${plyFile.length()}")
            Log.d(TAG, "Room bounds: ${maxX - minX}m x ${maxY - minY}m x ${maxZ - minZ}m min=($minX,$minY,$minZ) max=($maxX,$maxY,$maxZ)")

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
