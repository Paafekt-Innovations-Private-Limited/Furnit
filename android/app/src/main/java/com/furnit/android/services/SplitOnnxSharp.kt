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

            val sessionOptions = OrtSession.SessionOptions().apply {
                // Enable graph optimizations (BASIC is a good balance for mobile)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)

                // Use available cores for parallelism (capped at 4 to balance memory)
                val numCores = Runtime.getRuntime().availableProcessors()
                val intraThreads = minOf(numCores, 4)
                setIntraOpNumThreads(intraThreads)
                setInterOpNumThreads(1)
                Log.d(TAG, "Using $intraThreads intra-op threads (device has $numCores cores)")

                // Memory pattern helps reuse buffers, arena allocator uses too much memory
                setMemoryPatternOptimization(true)
                setCPUArenaAllocator(false)  // Arena pre-allocates too much memory
                setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)

                try {
                    // Keep mmap for the big split weights
                    addConfigEntry("session.use_mmap", "1")
                    // Keep prepacking enabled for GEMM performance
                    addConfigEntry("session.enable_mem_reuse", "1")
                } catch (e: Exception) {
                    Log.w(TAG, "Could not set session config: ${e.message}")
                }
            }

            progressCallback?.invoke(baseProgress + 0.02f, "Part $partNumber/4: Creating session...")
            Log.d(TAG, "Creating session for part $partNumber...")
            session = ortEnv.createSession(modelPath, sessionOptions)
            Log.d(TAG, "Session created for part $partNumber")
            progressCallback?.invoke(baseProgress + 0.05f, "Part $partNumber/4: Session ready")

            // Get input/output names
            val inputNames = session.inputNames.toList()
            val outputNames = session.outputNames.toList()
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

            val outputs = session.run(inputTensors)

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

            // Write float data in chunks to avoid large heap allocation
            val totalFloats = buffer.remaining()
            val chunkSize = 1024 * 1024  // 1M floats = 4MB per chunk
            val chunkBuffer = ByteBuffer.allocate(chunkSize * 4)
            chunkBuffer.order(ByteOrder.LITTLE_ENDIAN)

            var written = 0
            while (written < totalFloats) {
                val floatsToWrite = minOf(chunkSize, totalFloats - written)
                chunkBuffer.clear()
                chunkBuffer.limit(floatsToWrite * 4)

                val floatView = chunkBuffer.asFloatBuffer()
                for (i in 0 until floatsToWrite) {
                    floatView.put(buffer.get())
                }

                chunkBuffer.position(0)
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

            // Write float data in chunks to avoid large heap allocation
            val totalFloats = floatBuffer.remaining()
            val chunkSize = 1024 * 1024  // 1M floats = 4MB per chunk
            val chunkBuffer = ByteBuffer.allocate(chunkSize * 4)
            chunkBuffer.order(ByteOrder.LITTLE_ENDIAN)

            var written = 0
            while (written < totalFloats) {
                val floatsToWrite = minOf(chunkSize, totalFloats - written)
                chunkBuffer.clear()
                chunkBuffer.limit(floatsToWrite * 4)

                val floatView = chunkBuffer.asFloatBuffer()
                for (i in 0 until floatsToWrite) {
                    floatView.put(floatBuffer.get())
                }

                chunkBuffer.position(0)
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

            // Write PLY with batched I/O for performance
            val header = buildPlyHeader(gaussianCount)
            FileOutputStream(plyFile).buffered(256 * 1024).use { fos ->
                fos.write(header.toByteArray(Charsets.UTF_8))

                // Batch 500 vertices per write (~124KB batches)
                val batchSize = 500
                val batchBuffer = ByteBuffer.allocate(BYTES_PER_VERTEX * batchSize)
                batchBuffer.order(ByteOrder.LITTLE_ENDIAN)
                val scaleBoost = 1.3f
                val minScale = 0.001f

                var batchCount = 0
                for (i in 0 until gaussianCount) {
                    // Position
                    val posIdx = i * 3
                    val x = posBuffer.get(posIdx)
                    val y = -posBuffer.get(posIdx + 1)
                    val z = -posBuffer.get(posIdx + 2)

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
                    val colorIdx = i * 3
                    val r = colorBuffer.get(colorIdx).coerceIn(0f, 1f)
                    val g = colorBuffer.get(colorIdx + 1).coerceIn(0f, 1f)
                    val b = colorBuffer.get(colorIdx + 2).coerceIn(0f, 1f)
                    batchBuffer.putFloat((r - 0.5f) / SH_C0)
                    batchBuffer.putFloat((g - 0.5f) / SH_C0)
                    batchBuffer.putFloat((b - 0.5f) / SH_C0)

                    // Higher order SH (45 zeros)
                    for (j in 0 until 45) batchBuffer.putFloat(0f)

                    // Opacity -> logit
                    val rawOpacity = opacityBuffer.get(i).coerceIn(1e-4f, 1f - 1e-4f)
                    batchBuffer.putFloat(ln(rawOpacity / (1f - rawOpacity)))

                    // Scale -> log
                    val scaleIdx = i * 3
                    batchBuffer.putFloat(ln(max(scaleBuffer.get(scaleIdx) * scaleBoost, minScale)))
                    batchBuffer.putFloat(ln(max(scaleBuffer.get(scaleIdx + 1) * scaleBoost, minScale)))
                    batchBuffer.putFloat(ln(max(scaleBuffer.get(scaleIdx + 2) * scaleBoost, minScale)))

                    // Rotation -> normalize
                    val rotIdx = i * 4
                    val rw = rotBuffer.get(rotIdx)
                    val rx = rotBuffer.get(rotIdx + 1)
                    val ry = rotBuffer.get(rotIdx + 2)
                    val rz = rotBuffer.get(rotIdx + 3)
                    val mag = sqrt(rw * rw + rx * rx + ry * ry + rz * rz)
                    val invMag = if (mag > 1e-8f) 1f / mag else 1f
                    batchBuffer.putFloat(rw * invMag)
                    batchBuffer.putFloat(rx * invMag)
                    batchBuffer.putFloat(ry * invMag)
                    batchBuffer.putFloat(rz * invMag)

                    batchCount++
                    if (batchCount >= batchSize) {
                        fos.write(batchBuffer.array(), 0, batchBuffer.position())
                        batchBuffer.clear()
                        batchCount = 0
                    }
                }
                // Write remaining
                if (batchCount > 0) {
                    fos.write(batchBuffer.array(), 0, batchBuffer.position())
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
