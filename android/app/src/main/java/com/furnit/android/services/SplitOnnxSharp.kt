package com.furnit.android.services

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
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
        return PART_FILENAMES.all { (model, data) ->
            val modelFile = File(modelsDir, model)
            val dataFile = File(modelsDir, data)
            modelFile.exists() && dataFile.exists()
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
        progressCallback: ((Float, String) -> Unit)? = null
    ): StreamingResult? = withContext(Dispatchers.IO) {
        if (!isModelReady()) {
            Log.e(TAG, "Model parts not ready. Missing: ${getMissingFiles()}")
            progressCallback?.invoke(0f, "Model not ready")
            return@withContext null
        }

        try {
            val startTime = System.currentTimeMillis()
            progressCallback?.invoke(0.05f, "Initializing...")

            // Clean up any old temp files
            tempDir.listFiles()?.forEach { it.delete() }

            // Preprocess input image
            progressCallback?.invoke(0.1f, "Preprocessing image...")
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
            val inputBuffer = preprocessImageToBuffer(scaledBitmap)
            scaledBitmap.recycle()

            // Save input tensor to file for Part 1
            saveFloatBufferToFile(inputBuffer, File(tempDir, "input_image.tensor"), longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))

            var currentInputs: Map<String, File> = mapOf("image" to File(tempDir, "input_image.tensor"))

            // Run each part sequentially
            for (partIdx in 0 until NUM_PARTS) {
                // Better progress: each part gets 20% (total 80% for 4 parts)
                val partBaseProgress = 0.1f + (partIdx * 0.2f)
                val (modelFile, dataFile) = PART_FILENAMES[partIdx]
                progressCallback?.invoke(partBaseProgress, "Part ${partIdx + 1}/4: Loading model...")

                Log.d(TAG, "=== Part ${partIdx + 1} ===")
                Log.d(TAG, "Memory before: ${getMemoryInfo()}")
                Log.d(TAG, "Model file: $modelFile")

                // Request GC (no forced sleep - let system decide)
                System.gc()

                val modelPath = File(modelsDir, modelFile).absolutePath
                Log.d(TAG, "Loading from: $modelPath")
                val outputs = runModelPart(modelPath, currentInputs, partIdx + 1, progressCallback, partBaseProgress)

                if (outputs == null) {
                    Log.e(TAG, "Part ${partIdx + 1} failed")
                    return@withContext null
                }

                // IMPORTANT: Accumulate all outputs across parts - later parts may need
                // outputs from earlier parts (e.g., Part 3 needs weight tensors from Part 1)
                // Keep the original image tensor available for parts that need it
                val imageFile = File(tempDir, "input_image.tensor")
                currentInputs = currentInputs + outputs + ("image" to imageFile)
                Log.d(TAG, "Accumulated ${currentInputs.size} tensors for next part")

                Log.d(TAG, "Memory after Part ${partIdx + 1}: ${getMemoryInfo()}")
            }

            // Final part outputs are the Gaussian attributes
            progressCallback?.invoke(0.85f, "Processing Gaussian output...")

            val result = processGaussianOutput(currentInputs, progressCallback)

            // Clean up temp files
            tempDir.listFiles()?.forEach { it.delete() }

            val elapsed = System.currentTimeMillis() - startTime
            Log.d(TAG, "Split inference completed in ${elapsed}ms")
            progressCallback?.invoke(1.0f, "Done!")

            return@withContext result

        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OUT OF MEMORY during split inference", e)
            System.gc()
            progressCallback?.invoke(0f, "Out of memory")
            return@withContext null
        } catch (e: Exception) {
            Log.e(TAG, "Split inference failed", e)
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
        var session: OrtSession? = null

        try {
            // Try NNAPI first, fall back to CPU if it fails
            val createdSession = try {
                createSessionWithNnapi(modelPath, partNumber)
            } catch (e: Exception) {
                Log.w(TAG, "NNAPI failed for part $partNumber, falling back to CPU: ${e.message}")
                progressCallback?.invoke(baseProgress + 0.01f, "Part $partNumber/4: Using CPU fallback...")
                createSessionCpuOnly(modelPath, partNumber)
            }
            session = createdSession

            progressCallback?.invoke(baseProgress + 0.02f, "Part $partNumber/4: Session ready...")
            Log.d(TAG, "Session created for part $partNumber")
            progressCallback?.invoke(baseProgress + 0.05f, "Part $partNumber/4: Session ready")

            // Get input/output names
            val inputNames = createdSession.inputNames.toList()
            val outputNames = createdSession.outputNames.toList()
            Log.d(TAG, "Part $partNumber - Inputs: $inputNames, Outputs: $outputNames")

            // Load input tensors from files
            progressCallback?.invoke(baseProgress + 0.07f, "Part $partNumber/4: Loading ${inputNames.size} inputs...")
            val inputTensors = mutableMapOf<String, OnnxTensor>()
            for ((idx, name) in inputNames.withIndex()) {
                val file = inputs[name]
                if (file == null || !file.exists()) {
                    Log.e(TAG, "Missing input tensor: $name (available: ${inputs.keys})")
                    return null
                }
                val tensor = loadTensorFromFile(ortEnv, file, name)
                if (tensor != null) {
                    inputTensors[name] = tensor
                } else {
                    Log.e(TAG, "Failed to load tensor: $name from ${file.absolutePath}")
                    return null
                }
                if (idx % 10 == 0) {
                    Log.d(TAG, "Loaded ${idx + 1}/${inputNames.size} inputs")
                }
            }
            Log.d(TAG, "All ${inputNames.size} inputs loaded for part $partNumber")

            // Run inference
            progressCallback?.invoke(baseProgress + 0.10f, "Part $partNumber/4: Running inference...")
            Log.d(TAG, "Running Part $partNumber inference with ${inputTensors.size} inputs...")
            val inferStartTime = System.currentTimeMillis()

            val outputs = createdSession.run(inputTensors)

            val inferTime = System.currentTimeMillis() - inferStartTime
            Log.d(TAG, "Part $partNumber inference completed in ${inferTime}ms")
            progressCallback?.invoke(baseProgress + 0.15f, "Part $partNumber/4: Inference done (${inferTime/1000}s)")

            // Save output tensors to files
            val outputFiles = mutableMapOf<String, File>()
            for (outputName in outputNames) {
                val tensor = outputs[outputName].get() as? OnnxTensor
                if (tensor != null) {
                    val file = File(tempDir, "part${partNumber}_${outputName.replace("/", "_")}.tensor")
                    saveTensorToFile(tensor, file)
                    outputFiles[outputName] = file
                    Log.d(TAG, "Saved output: $outputName -> ${file.name} (${file.length() / 1024}KB)")
                }
            }

            // Clean up
            inputTensors.values.forEach { it.close() }
            outputs.close()

            return outputFiles

        } catch (e: Exception) {
            Log.e(TAG, "Error running part $partNumber: ${e.message}", e)
            Log.e(TAG, "Stack trace: ${e.stackTraceToString()}")
            return null
        } finally {
            session?.close()
            // Note: ortEnv is reused across parts, don't close it

            // Force GC after closing session
            System.gc()
        }
    }

    /**
     * Create session with NNAPI acceleration (may throw if device doesn't support model ops).
     */
    private fun createSessionWithNnapi(modelPath: String, partNumber: Int): OrtSession {
        val sessionOptions = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)

            val numCores = Runtime.getRuntime().availableProcessors()
            val intraThreads = minOf(numCores, 4)
            setIntraOpNumThreads(intraThreads)
            setInterOpNumThreads(1)
            Log.d(TAG, "NNAPI: Using $intraThreads intra-op threads (device has $numCores cores)")

            setMemoryPatternOptimization(true)
            setCPUArenaAllocator(false)
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)

            try {
                addConfigEntry("session.use_mmap", "1")
                addConfigEntry("session.enable_mem_reuse", "1")
            } catch (e: Exception) {
                Log.w(TAG, "Could not set session config: ${e.message}")
            }

            // NNAPI: let unsupported ops fall back to CPU (no CPU_DISABLED)
            addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
            Log.d(TAG, "NNAPI EP registered (FP16, CPU fallback allowed) for part $partNumber")
        }

        Log.d(TAG, "Creating NNAPI session for part $partNumber...")
        return ortEnv.createSession(modelPath, sessionOptions)
    }

    /**
     * Create session with CPU-only execution (fallback when NNAPI fails).
     */
    private fun createSessionCpuOnly(modelPath: String, partNumber: Int): OrtSession {
        val sessionOptions = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)

            val numCores = Runtime.getRuntime().availableProcessors()
            val intraThreads = minOf(numCores, 4)
            setIntraOpNumThreads(intraThreads)
            setInterOpNumThreads(1)
            Log.d(TAG, "CPU: Using $intraThreads intra-op threads")

            setMemoryPatternOptimization(true)
            setCPUArenaAllocator(false)  // Arena pre-allocates too much for 2.4GB model
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)

            try {
                addConfigEntry("session.use_mmap", "1")
                addConfigEntry("session.enable_mem_reuse", "1")
            } catch (e: Exception) {
                Log.w(TAG, "Could not set session config: ${e.message}")
            }
        }

        Log.d(TAG, "Creating CPU-only session for part $partNumber...")
        return ortEnv.createSession(modelPath, sessionOptions)
    }

    /**
     * Preprocess image to float buffer.
     */
    private fun preprocessImageToBuffer(bitmap: Bitmap): FloatBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val floatBuffer = FloatBuffer.allocate(3 * width * height)

        // R channel
        for (pixel in pixels) {
            floatBuffer.put(((pixel shr 16) and 0xFF) / 255f)
        }
        // G channel
        for (pixel in pixels) {
            floatBuffer.put(((pixel shr 8) and 0xFF) / 255f)
        }
        // B channel
        for (pixel in pixels) {
            floatBuffer.put((pixel and 0xFF) / 255f)
        }

        floatBuffer.rewind()
        return floatBuffer
    }

    /**
     * Save a FloatBuffer to file with shape metadata using streaming.
     */
    private fun saveFloatBufferToFile(buffer: FloatBuffer, file: File, shape: LongArray) {
        buffer.rewind()

        RandomAccessFile(file, "rw").use { raf ->
            val channel = raf.channel

            // Write shape metadata (num dims + dims)
            val headerSize = 4 + shape.size * 8
            val headerBuffer = ByteBuffer.allocate(headerSize)
            headerBuffer.order(ByteOrder.LITTLE_ENDIAN)
            headerBuffer.putInt(shape.size)
            shape.forEach { headerBuffer.putLong(it) }
            headerBuffer.flip()
            channel.write(headerBuffer)

            // Write float data in chunks using DirectByteBuffer for zero-copy to disk
            val totalFloats = buffer.remaining()
            val chunkSize = 1024 * 1024  // 1M floats = 4MB per chunk
            val chunkBuffer = ByteBuffer.allocateDirect(chunkSize * 4)
            chunkBuffer.order(ByteOrder.LITTLE_ENDIAN)
            val tempArray = FloatArray(chunkSize)

            var written = 0
            while (written < totalFloats) {
                val floatsToWrite = minOf(chunkSize, totalFloats - written)

                // Bulk read into local array, then bulk write to direct buffer
                buffer.get(tempArray, 0, floatsToWrite)
                chunkBuffer.clear()
                chunkBuffer.asFloatBuffer().put(tempArray, 0, floatsToWrite)
                chunkBuffer.position(0)
                chunkBuffer.limit(floatsToWrite * 4)
                channel.write(chunkBuffer)
                written += floatsToWrite
            }
        }
    }

    /**
     * Save an ONNX tensor to file using streaming (avoids large heap allocations).
     */
    private fun saveTensorToFile(tensor: OnnxTensor, file: File) {
        val shape = tensor.info.shape
        val floatBuffer = tensor.floatBuffer
        floatBuffer.rewind()

        RandomAccessFile(file, "rw").use { raf ->
            val channel = raf.channel

            // Write shape metadata
            val headerSize = 4 + shape.size * 8
            val headerBuffer = ByteBuffer.allocate(headerSize)
            headerBuffer.order(ByteOrder.LITTLE_ENDIAN)
            headerBuffer.putInt(shape.size)
            shape.forEach { headerBuffer.putLong(it) }
            headerBuffer.flip()
            channel.write(headerBuffer)

            // Write float data in chunks using DirectByteBuffer for zero-copy to disk
            val totalFloats = floatBuffer.remaining()
            val chunkSize = 1024 * 1024  // 1M floats = 4MB per chunk
            val chunkBuffer = ByteBuffer.allocateDirect(chunkSize * 4)
            chunkBuffer.order(ByteOrder.LITTLE_ENDIAN)
            val tempArray = FloatArray(chunkSize)

            var written = 0
            while (written < totalFloats) {
                val floatsToWrite = minOf(chunkSize, totalFloats - written)

                // Bulk read into local array, then bulk write to direct buffer
                floatBuffer.get(tempArray, 0, floatsToWrite)
                chunkBuffer.clear()
                chunkBuffer.asFloatBuffer().put(tempArray, 0, floatsToWrite)
                chunkBuffer.position(0)
                chunkBuffer.limit(floatsToWrite * 4)
                channel.write(chunkBuffer)
                written += floatsToWrite
            }
        }
    }

    /**
     * Load a tensor from file.
     */
    private fun loadTensorFromFile(env: OrtEnvironment, file: File, name: String): OnnxTensor? {
        try {
            RandomAccessFile(file, "r").use { raf ->
                val channel = raf.channel

                // Read shape metadata
                val numDimsBuffer = ByteBuffer.allocate(4)
                numDimsBuffer.order(ByteOrder.LITTLE_ENDIAN)
                channel.read(numDimsBuffer)
                numDimsBuffer.flip()
                val numDims = numDimsBuffer.int

                val shapeBuffer = ByteBuffer.allocate(numDims * 8)
                shapeBuffer.order(ByteOrder.LITTLE_ENDIAN)
                channel.read(shapeBuffer)
                shapeBuffer.flip()
                val shape = LongArray(numDims) { shapeBuffer.long }

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
            Log.e(TAG, "Failed to load tensor from $file", e)
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
        try {
            // Find the output files (named with part4_ prefix)
            val positionsFile = outputs.entries.find { it.key.contains("positions") }?.value
            val scalesFile = outputs.entries.find { it.key.contains("scales") }?.value
            val rotationsFile = outputs.entries.find { it.key.contains("rotations") }?.value
            val colorsFile = outputs.entries.find { it.key.contains("colors") }?.value
            val opacityFile = outputs.entries.find { it.key.contains("opacity") }?.value

            if (positionsFile == null || scalesFile == null || rotationsFile == null ||
                colorsFile == null || opacityFile == null) {
                Log.e(TAG, "Missing output tensors. Available: ${outputs.keys}")
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

                        // Higher order SH (45 zeros)
                        repeat(45) { batchBuffer.putFloat(0f) }

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

            Log.d(TAG, "PLY written: $gaussianCount Gaussians")
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
            Log.e(TAG, "Failed to process Gaussian output", e)
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
