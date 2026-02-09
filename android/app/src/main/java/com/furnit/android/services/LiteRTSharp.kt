package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/**
 * SHARP Gaussian Splatting using LiteRT (TFLite).
 *
 * **4-part single-patch mode** (preferred):
 *   Parts 1 & 2 process one 384x384 patch at a time (looped 35x on Android)
 *   to avoid the ~750MB attention tensor from batching 35 patches.
 *
 *   Part 1: Single-Patch Encoder A (ViT blocks 0-11) ~290MB — run 35x
 *   Part 2: Single-Patch Encoder B (ViT blocks 12-23) ~288MB — run 35x
 *   Part 3: Image Encoder A (ViT blocks 0-11) ~291MB — run 1x
 *   Part 4: Image Encoder B + Full Decoder + Gaussians ~387MB — run 1x
 *
 * Pipeline:
 *   1. Preprocess image → [1,3,1536,1536] in [0,1]
 *   2. Create pyramid: x0(1536²), x1(768²), x2(384²)
 *   3. Split into 35 overlapping 384x384 patches (25+9+1)
 *   4. Load Part 1, run on each patch → tokens[i], block5[i] — save to disk
 *   5. Load Part 2, run on each tokens[i] → features[i] — save to disk
 *   6. Reshape block5/tokens (remove CLS, reshape to [1024,24,24])
 *   7. Merge patches → latent0, latent1, x0_feat, x1_feat, x2_feat
 *   8. Run Part 3 on full image → image_tokens
 *   9. Run Part 4 → packed [1,N,14] Gaussians → write PLY
 *
 * Output layout per Gaussian (14 floats):
 *   [0-2]   position xyz
 *   [3]     opacity (raw logit → apply sigmoid)
 *   [4-6]   scale (singular values)
 *   [7-10]  rotation quaternion wxyz
 *   [11-13] color RGB
 */
class LiteRTSharp private constructor(private val context: Context) {

    companion object {
        private const val TAG = "LiteRTSharp"
        private const val IMAGE_SIZE = 1536
        private const val PARAMS_PER_GAUSSIAN = 14
        private const val SH_C0 = 0.28209479177387814f
        private const val BYTES_PER_VERTEX = 62 * 4

        // 4-part split model filenames
        private const val NUM_SPLIT_PARTS = 4
        private val SPLIT_PART_FILENAMES = arrayOf(
            "sharp_part1_fp16.tflite",
            "sharp_part2_fp16.tflite",
            "sharp_part3_fp16.tflite",
            "sharp_part4_fp16.tflite",
        )

        // Sliding pyramid patch configuration
        private const val PATCH_SIZE = 384
        private const val NUM_PATCHES = 35       // 25 (x0) + 9 (x1) + 1 (x2)
        private const val X0_PATCH_COUNT = 25    // 5x5 grid
        private const val X1_PATCH_COUNT = 9     // 3x3 grid
        private const val FEATURE_DIM = 1024
        private const val GRID_H = 24
        private const val GRID_W = 24
        private const val NUM_PREFIX_TOKENS = 1  // CLS token
        private const val X0_MERGE_PADDING = 3
        private const val X1_MERGE_PADDING = 6

        // Single model filenames (fallback)
        private val SINGLE_MODEL_FILENAMES = arrayOf(
            "vit_gaussian_fp16.tflite",
            "vit_gaussian_fp32.tflite",
        )

        private val EXTRA_SEARCH_DIRS = arrayOf(
            "/data/local/tmp/furnit/",
        )

        // Pre-computed logit LUT: maps opacity [0,1] → ln(p/(1-p))
        private const val LOGIT_LUT_SIZE = 1024
        private val LOGIT_LUT = FloatArray(LOGIT_LUT_SIZE) { i ->
            val p = (i.toFloat() / (LOGIT_LUT_SIZE - 1)).coerceIn(1e-4f, 1f - 1e-4f)
            ln(p / (1f - p))
        }

        // Pre-computed natural log LUT for scale transform
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
        private var instance: LiteRTSharp? = null

        fun getInstance(context: Context): LiteRTSharp {
            return instance ?: synchronized(this) {
                instance ?: LiteRTSharp(context.applicationContext).also {
                    instance = it
                    Log.d(TAG, "LiteRTSharp singleton created")
                }
            }
        }
    }

    private var isInitialized = false
    private var useSplitMode = false

    private val modelsDir: File by lazy {
        context.getExternalFilesDir("models") ?: File(context.filesDir, "models")
    }

    private val tempDir: File by lazy {
        File(context.cacheDir, "litert_temp").also { it.mkdirs() }
    }

    data class StreamingResult(
        val plyFile: File,
        val classicPlyFile: File,
        val gaussianCount: Int,
        val roomWidth: Float,
        val roomHeight: Float,
        val roomDepth: Float
    )

    private fun findFile(filename: String): File? {
        val inModelsDir = File(modelsDir, filename)
        if (inModelsDir.exists() && inModelsDir.length() > 0) return inModelsDir

        for (dir in EXTRA_SEARCH_DIRS) {
            val file = File(dir, filename)
            if (file.exists() && file.length() > 0) return file
        }
        return null
    }

    private fun findSingleModel(): File? {
        for (filename in SINGLE_MODEL_FILENAMES) {
            val file = findFile(filename)
            if (file != null) return file
        }
        return null
    }

    private fun isSplitModelReady(): Boolean {
        return SPLIT_PART_FILENAMES.all { findFile(it) != null }
    }

    fun isModelReady(): Boolean = isSplitModelReady() || findSingleModel() != null

    fun initialize(): Boolean {
        // Prefer 4-part split mode (avoids OOM)
        if (isSplitModelReady()) {
            Log.d(TAG, "4-part split mode:")
            for (filename in SPLIT_PART_FILENAMES) {
                val file = findFile(filename)!!
                Log.d(TAG, "  ${file.name}: ${file.length() / 1024 / 1024}MB")
            }
            useSplitMode = true
            isInitialized = true
            return true
        }

        // Fallback to single model
        val modelFile = findSingleModel()
        if (modelFile == null) {
            Log.e(TAG, "No model found (tried split: ${SPLIT_PART_FILENAMES.joinToString()}, single: ${SINGLE_MODEL_FILENAMES.joinToString()})")
            return false
        }

        Log.d(TAG, "Single mode: ${modelFile.absolutePath} (${modelFile.length() / 1024 / 1024}MB)")
        useSplitMode = false
        isInitialized = true
        return true
    }

    /**
     * Create a TFLite Interpreter. For large models (>500MB), use CPU-only with
     * XNNPACK to avoid OOM from GPU delegate copying weights to GPU memory.
     *
     * GPU delegate is only attempted if CompatibilityList reports support AND the
     * model is under 500MB. If interpreter creation with GPU fails (native crash /
     * unsupported ops), we transparently retry CPU-only.
     */
    private fun createInterpreter(modelFile: File): Interpreter {
        val modelSizeMB = modelFile.length() / 1024 / 1024
        val numThreads = Runtime.getRuntime().availableProcessors()

        // Try GPU delegate for small-enough models on supported devices
        if (modelSizeMB < 500) {
            var gpuDelegate: GpuDelegate? = null
            try {
                val compatList = CompatibilityList()
                if (compatList.isDelegateSupportedOnThisDevice) {
                    val gpuOptions = compatList.bestOptionsForThisDevice
                    gpuDelegate = GpuDelegate(gpuOptions)
                    val gpuInterpreterOptions = Interpreter.Options().apply {
                        setNumThreads(numThreads)
                        addDelegate(gpuDelegate)
                    }
                    val interpreter = Interpreter(modelFile, gpuInterpreterOptions)
                    Log.d(TAG, "GPU delegate enabled for ${modelFile.name}")
                    return interpreter
                } else {
                    Log.w(TAG, "GPU delegate not supported on this device — using CPU")
                }
            } catch (e: Exception) {
                Log.w(TAG, "GPU delegate failed for ${modelFile.name}, falling back to CPU: ${e.message}")
                try { gpuDelegate?.close() } catch (_: Exception) {}
            } catch (e: Error) {
                // Catches UnsatisfiedLinkError, native crashes surfaced as java.lang.Error
                Log.w(TAG, "GPU delegate native error for ${modelFile.name}, falling back to CPU: ${e.message}")
                try { gpuDelegate?.close() } catch (_: Exception) {}
            }
        }

        // CPU-only fallback (or forced for large models)
        val cpuOptions = Interpreter.Options().apply {
            setNumThreads(numThreads)
            setAllowFp16PrecisionForFp32(true)
        }
        Log.d(TAG, "CPU mode for ${modelFile.name} (${modelSizeMB}MB)")
        return Interpreter(modelFile, cpuOptions)
    }

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
     * Run SHARP inference — delegates to split or single mode based on available models.
     */
    suspend fun inferStreaming(
        bitmap: Bitmap,
        progressCallback: ((Float, String) -> Unit)? = null
    ): StreamingResult? = withContext(Dispatchers.IO) {
        if (!isInitialized) {
            Log.e(TAG, "Not initialized")
            return@withContext null
        }

        try {
            if (useSplitMode) {
                inferSplitMode(bitmap, progressCallback)
            } else {
                inferSingleMode(bitmap, progressCallback)
            }
        } catch (e: Exception) {
            Log.e(TAG, "LiteRT inference failed: ${e.message}", e)
            return@withContext null
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "LiteRT OOM during inference: ${e.message}")
            System.gc()
            return@withContext null
        }
    }

    // ========================================================================
    // 4-part single-patch mode: pyramid → loop Parts 1&2 × 35 → merge → Parts 3&4
    // ========================================================================

    /**
     * Extract overlapping 384x384 patches from a CHW float buffer.
     * Returns list of [1, 3, 384, 384] ByteBuffers.
     */
    private fun extractPatches(
        imageBuffer: ByteBuffer,
        imageSize: Int,
        overlapRatio: Float
    ): List<ByteBuffer> {
        val patchStride = (PATCH_SIZE * (1f - overlapRatio)).toInt()
        val steps = ((imageSize - PATCH_SIZE).toFloat() / patchStride).toInt() + 1
        val channelSize = imageSize * imageSize

        imageBuffer.rewind()
        val srcFloats = FloatArray(3 * channelSize)
        imageBuffer.asFloatBuffer().get(srcFloats)

        val patches = mutableListOf<ByteBuffer>()
        val patchFloatSize = 3 * PATCH_SIZE * PATCH_SIZE
        for (j in 0 until steps) {
            val j0 = j * patchStride
            for (i in 0 until steps) {
                val i0 = i * patchStride
                val patchBuf = ByteBuffer.allocateDirect(patchFloatSize * 4)
                    .apply { order(ByteOrder.nativeOrder()) }
                val fb = patchBuf.asFloatBuffer()
                for (c in 0..2) {
                    val chanOffset = c * channelSize
                    for (row in 0 until PATCH_SIZE) {
                        val srcIdx = chanOffset + (j0 + row) * imageSize + i0
                        fb.put(srcFloats, srcIdx, PATCH_SIZE)
                    }
                }
                patchBuf.rewind()
                patches.add(patchBuf)
            }
        }
        return patches
    }

    /**
     * Resize a CHW [1,3,H,W] image buffer using bilinear interpolation.
     */
    private fun resizeImageBuffer(src: ByteBuffer, srcSize: Int, dstSize: Int): ByteBuffer {
        src.rewind()
        val srcFloats = FloatArray(3 * srcSize * srcSize)
        src.asFloatBuffer().get(srcFloats)

        val dstFloats = FloatArray(3 * dstSize * dstSize)
        val scale = srcSize.toFloat() / dstSize

        for (c in 0..2) {
            val srcChanOff = c * srcSize * srcSize
            val dstChanOff = c * dstSize * dstSize
            for (dy in 0 until dstSize) {
                val sy = dy * scale
                val sy0 = sy.toInt().coerceIn(0, srcSize - 2)
                val sy1 = sy0 + 1
                val fy = sy - sy0
                for (dx in 0 until dstSize) {
                    val sx = dx * scale
                    val sx0 = sx.toInt().coerceIn(0, srcSize - 2)
                    val sx1 = sx0 + 1
                    val fx = sx - sx0
                    val v00 = srcFloats[srcChanOff + sy0 * srcSize + sx0]
                    val v01 = srcFloats[srcChanOff + sy0 * srcSize + sx1]
                    val v10 = srcFloats[srcChanOff + sy1 * srcSize + sx0]
                    val v11 = srcFloats[srcChanOff + sy1 * srcSize + sx1]
                    dstFloats[dstChanOff + dy * dstSize + dx] =
                        (1 - fy) * ((1 - fx) * v00 + fx * v01) + fy * ((1 - fx) * v10 + fx * v11)
                }
            }
        }

        val buf = ByteBuffer.allocateDirect(dstFloats.size * 4).apply { order(ByteOrder.nativeOrder()) }
        buf.asFloatBuffer().put(dstFloats)
        buf.rewind()
        return buf
    }

    /**
     * Reshape tokens [1, 577, 1024] → spatial [1, 1024, 24, 24] by removing CLS and reshaping.
     * Returns a float array of size 1024*24*24.
     */
    private fun reshapeTokensToSpatial(tokensFile: File): FloatArray {
        val buf = loadTensorFromFile(tokensFile)
        val floats = FloatArray(577 * FEATURE_DIM)
        buf.asFloatBuffer().get(floats)

        // Remove CLS token (first token), keep [576, 1024]
        // Reshape [576, 1024] = [24, 24, 1024] → permute to [1024, 24, 24]
        val spatial = FloatArray(FEATURE_DIM * GRID_H * GRID_W)
        for (h in 0 until GRID_H) {
            for (w in 0 until GRID_W) {
                val seqIdx = NUM_PREFIX_TOKENS + h * GRID_W + w  // skip CLS
                for (c in 0 until FEATURE_DIM) {
                    spatial[c * GRID_H * GRID_W + h * GRID_W + w] = floats[seqIdx * FEATURE_DIM + c]
                }
            }
        }
        return spatial
    }

    /**
     * Merge a grid of spatial patches [1024,24,24] into one feature map, trimming overlap.
     * Returns a ByteBuffer with the merged [1, 1024, outH, outW] tensor.
     */
    private fun mergePatches(
        patchSpatials: List<FloatArray>,
        gridSteps: Int,
        padding: Int
    ): ByteBuffer {
        val patchSpatialSize = GRID_H  // 24
        // Calculate output size per row/col after trimming
        val trimmedSize = patchSpatialSize - 2 * padding  // inner patches
        val outputSize = gridSteps * trimmedSize + 2 * padding  // corners keep padding on edges
        // Actually: first/last row/col keep padding on outer edge, inner patches trimmed both sides
        // Simpler: compute per-patch contribution
        val outH = patchSpatialSize + (gridSteps - 1) * (patchSpatialSize - 2 * padding)
        val outW = outH

        val merged = FloatArray(FEATURE_DIM * outH * outW)

        var idx = 0
        for (j in 0 until gridSteps) {
            for (i in 0 until gridSteps) {
                val patch = patchSpatials[idx]
                val rowStart = if (j != 0) padding else 0
                val rowEnd = if (j != gridSteps - 1) patchSpatialSize - padding else patchSpatialSize
                val colStart = if (i != 0) padding else 0
                val colEnd = if (i != gridSteps - 1) patchSpatialSize - padding else patchSpatialSize

                // Destination offset in output grid
                val dstRowOff = if (j == 0) 0 else padding + j * (patchSpatialSize - 2 * padding)
                val dstColOff = if (i == 0) 0 else padding + i * (patchSpatialSize - 2 * padding)

                for (c in 0 until FEATURE_DIM) {
                    for (r in rowStart until rowEnd) {
                        for (col in colStart until colEnd) {
                            val srcIdx = c * patchSpatialSize * patchSpatialSize + r * patchSpatialSize + col
                            val dstR = dstRowOff + (r - rowStart)
                            val dstC = dstColOff + (col - colStart)
                            merged[c * outH * outW + dstR * outW + dstC] = patch[srcIdx]
                        }
                    }
                }
                idx++
            }
        }

        val numElements = FEATURE_DIM * outH * outW
        val buf = ByteBuffer.allocateDirect(numElements * 4).apply { order(ByteOrder.nativeOrder()) }
        buf.asFloatBuffer().put(merged)
        buf.rewind()

        // Save with shape metadata
        return buf
    }

    /**
     * Save a FloatArray as a tensor file with shape metadata.
     */
    private fun saveMergedTensor(data: ByteBuffer, shape: IntArray, file: File) {
        saveTensorToFile(data, shape, file)
    }

    /**
     * Read the shape header from a tensor file without loading the data.
     */
    private fun readTensorShape(file: File): IntArray {
        RandomAccessFile(file, "r").use { raf ->
            val channel = raf.channel
            val numDimsBuf = ByteBuffer.allocate(4).apply { order(ByteOrder.LITTLE_ENDIAN) }
            channel.read(numDimsBuf)
            numDimsBuf.flip()
            val numDims = numDimsBuf.int
            val shapeBuf = ByteBuffer.allocate(numDims * 8).apply { order(ByteOrder.LITTLE_ENDIAN) }
            channel.read(shapeBuf)
            shapeBuf.flip()
            return IntArray(numDims) { shapeBuf.long.toInt() }
        }
    }

    private suspend fun inferSplitMode(
        bitmap: Bitmap,
        progressCallback: ((Float, String) -> Unit)?
    ): StreamingResult? {
        try {
            val startTime = System.currentTimeMillis()
            Log.d(TAG, "=== Single-Patch LiteRT inference ===")
            Log.d(TAG, "Memory before: ${getMemoryInfo()}")

            tempDir.listFiles()?.forEach { it.delete() }

            // Step 1: Preprocess image to [1,3,1536,1536] in [0,1]
            progressCallback?.invoke(0.02f, "Preprocessing image...")
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
            val inputBuffer = preprocessImage(scaledBitmap)
            scaledBitmap.recycle()
            Log.d(TAG, "Image preprocessed to ${IMAGE_SIZE}x${IMAGE_SIZE}")

            // Step 2: Create pyramid and extract patches
            progressCallback?.invoke(0.03f, "Creating pyramid patches...")
            val x1Buffer = resizeImageBuffer(inputBuffer, IMAGE_SIZE, IMAGE_SIZE / 2)  // 768
            val x2Buffer = resizeImageBuffer(inputBuffer, IMAGE_SIZE, IMAGE_SIZE / 4)  // 384

            val x0Patches = extractPatches(inputBuffer, IMAGE_SIZE, 0.25f)      // 25 patches
            val x1Patches = extractPatches(x1Buffer, IMAGE_SIZE / 2, 0.5f)      // 9 patches
            val x2Patches = listOf(x2Buffer)                                      // 1 patch
            val allPatches = x0Patches + x1Patches + x2Patches
            Log.d(TAG, "Patches: ${x0Patches.size} (x0) + ${x1Patches.size} (x1) + 1 (x2) = ${allPatches.size}")

            // Step 3: Load Part 1, run on each patch
            progressCallback?.invoke(0.05f, "Part 1: Encoding patches (0/${allPatches.size})...")
            val part1File = findFile(SPLIT_PART_FILENAMES[0])!!
            Log.d(TAG, "Loading Part 1: ${part1File.name} (${part1File.length() / 1024 / 1024}MB)")
            var interpreter = createInterpreter(part1File)

            val tokensShape = interpreter.getOutputTensor(0).shape()
            val tokensSize = tokensShape.fold(1) { acc, d -> acc * d }
            val block5Shape = interpreter.getOutputTensor(1).shape()
            val block5Size = block5Shape.fold(1) { acc, d -> acc * d }

            val part1Start = System.currentTimeMillis()
            for (i in allPatches.indices) {
                if (i % 5 == 0) {
                    val pct = 0.05f + (i.toFloat() / allPatches.size) * 0.25f
                    progressCallback?.invoke(pct, "Part 1: Encoding patches ($i/${allPatches.size})...")
                }

                allPatches[i].rewind()
                val tokensBuf = ByteBuffer.allocateDirect(tokensSize * 4).apply { order(ByteOrder.nativeOrder()) }
                val block5Buf = ByteBuffer.allocateDirect(block5Size * 4).apply { order(ByteOrder.nativeOrder()) }
                val outputMap = HashMap<Int, Any>()
                outputMap[0] = tokensBuf
                outputMap[1] = block5Buf
                interpreter.runForMultipleInputsOutputs(arrayOf(allPatches[i]), outputMap)

                saveTensorToFile(tokensBuf, tokensShape, File(tempDir, "tokens_$i.tensor"))
                saveTensorToFile(block5Buf, block5Shape, File(tempDir, "block5_$i.tensor"))
            }
            interpreter.close()
            System.gc()
            val part1Time = System.currentTimeMillis() - part1Start
            Log.d(TAG, "Part 1 done: ${allPatches.size} patches in ${part1Time}ms (${part1Time / allPatches.size}ms/patch)")

            // Step 4: Load Part 2, run on each token set
            progressCallback?.invoke(0.30f, "Part 2: Processing features (0/${allPatches.size})...")
            val part2File = findFile(SPLIT_PART_FILENAMES[1])!!
            Log.d(TAG, "Loading Part 2: ${part2File.name} (${part2File.length() / 1024 / 1024}MB)")
            interpreter = createInterpreter(part2File)

            val featShape = interpreter.getOutputTensor(0).shape()
            val featSize = featShape.fold(1) { acc, d -> acc * d }

            val part2Start = System.currentTimeMillis()
            for (i in allPatches.indices) {
                if (i % 5 == 0) {
                    val pct = 0.30f + (i.toFloat() / allPatches.size) * 0.15f
                    progressCallback?.invoke(pct, "Part 2: Processing features ($i/${allPatches.size})...")
                }

                val tokensInput = loadTensorFromFile(File(tempDir, "tokens_$i.tensor"))
                val featBuf = ByteBuffer.allocateDirect(featSize * 4).apply { order(ByteOrder.nativeOrder()) }
                val outputMap = HashMap<Int, Any>()
                outputMap[0] = featBuf
                interpreter.runForMultipleInputsOutputs(arrayOf(tokensInput), outputMap)

                saveTensorToFile(featBuf, featShape, File(tempDir, "feat_$i.tensor"))
            }
            interpreter.close()
            System.gc()
            val part2Time = System.currentTimeMillis() - part2Start
            Log.d(TAG, "Part 2 done: ${allPatches.size} patches in ${part2Time}ms (${part2Time / allPatches.size}ms/patch)")

            // Step 5: Reshape and merge patches on CPU
            progressCallback?.invoke(0.45f, "Merging patch features...")
            val mergeStart = System.currentTimeMillis()

            // Reshape block5 tokens → spatial [1024,24,24] for first 25 patches (x0 only → latent0)
            val block5Spatials = (0 until X0_PATCH_COUNT).map {
                reshapeTokensToSpatial(File(tempDir, "block5_$it.tensor"))
            }
            val latent0Buf = mergePatches(block5Spatials, 5, X0_MERGE_PADDING)
            val latent0OutH = GRID_H + 4 * (GRID_H - 2 * X0_MERGE_PADDING)  // 24 + 4*18 = 96
            saveTensorToFile(latent0Buf, intArrayOf(1, FEATURE_DIM, latent0OutH, latent0OutH),
                File(tempDir, "latent0.tensor"))

            // Reshape tokens (=block11) → spatial for first 25 patches → latent1
            val block11Spatials = (0 until X0_PATCH_COUNT).map {
                reshapeTokensToSpatial(File(tempDir, "tokens_$it.tensor"))
            }
            val latent1Buf = mergePatches(block11Spatials, 5, X0_MERGE_PADDING)
            saveTensorToFile(latent1Buf, intArrayOf(1, FEATURE_DIM, latent0OutH, latent0OutH),
                File(tempDir, "latent1.tensor"))

            // Merge Part 2 features
            // x0: first 25 → 5x5 grid, padding=3
            val x0FeatureSpatials = (0 until X0_PATCH_COUNT).map {
                val buf = loadTensorFromFile(File(tempDir, "feat_$it.tensor"))
                val floats = FloatArray(featSize)
                buf.asFloatBuffer().get(floats)
                floats
            }
            val x0FeatBuf = mergePatches(x0FeatureSpatials, 5, X0_MERGE_PADDING)
            saveTensorToFile(x0FeatBuf, intArrayOf(1, FEATURE_DIM, latent0OutH, latent0OutH),
                File(tempDir, "x0_feat.tensor"))

            // x1: patches 25-33 → 3x3 grid, padding=6
            val x1FeatureSpatials = (X0_PATCH_COUNT until X0_PATCH_COUNT + X1_PATCH_COUNT).map {
                val buf = loadTensorFromFile(File(tempDir, "feat_$it.tensor"))
                val floats = FloatArray(featSize)
                buf.asFloatBuffer().get(floats)
                floats
            }
            val x1OutH = GRID_H + 2 * (GRID_H - 2 * X1_MERGE_PADDING)  // 24 + 2*12 = 48
            val x1FeatBuf = mergePatches(x1FeatureSpatials, 3, X1_MERGE_PADDING)
            saveTensorToFile(x1FeatBuf, intArrayOf(1, FEATURE_DIM, x1OutH, x1OutH),
                File(tempDir, "x1_feat.tensor"))

            // x2: patch 34, no merging needed — just copy feat_34
            File(tempDir, "feat_34.tensor").copyTo(File(tempDir, "x2_feat.tensor"), overwrite = true)

            // Clean up individual patch files
            for (i in 0 until NUM_PATCHES) {
                File(tempDir, "tokens_$i.tensor").delete()
                File(tempDir, "block5_$i.tensor").delete()
                File(tempDir, "feat_$i.tensor").delete()
            }
            System.gc()
            val mergeTime = System.currentTimeMillis() - mergeStart
            Log.d(TAG, "Merge done in ${mergeTime}ms. Memory: ${getMemoryInfo()}")

            // Step 6: Part 3 — Image Encoder A (full image → image_tokens)
            progressCallback?.invoke(0.50f, "Part 3: Image encoder...")
            val part3File = findFile(SPLIT_PART_FILENAMES[2])!!
            Log.d(TAG, "Loading Part 3: ${part3File.name}")
            interpreter = createInterpreter(part3File)
            val imgTokensShape = interpreter.getOutputTensor(0).shape()
            val imgTokensSize = imgTokensShape.fold(1) { acc, d -> acc * d }
            val imgTokensBuf = ByteBuffer.allocateDirect(imgTokensSize * 4).apply { order(ByteOrder.nativeOrder()) }

            inputBuffer.rewind()
            val part3Start = System.currentTimeMillis()
            interpreter.run(inputBuffer, imgTokensBuf)
            val part3Time = System.currentTimeMillis() - part3Start
            Log.d(TAG, "Part 3 done in ${part3Time}ms")

            saveTensorToFile(imgTokensBuf, imgTokensShape, File(tempDir, "image_tokens.tensor"))
            interpreter.close()
            System.gc()

            // Step 7: Part 4 — Image Encoder B + Full Decoder + Gaussians
            progressCallback?.invoke(0.60f, "Part 4: Decoding Gaussians...")
            val part4File = findFile(SPLIT_PART_FILENAMES[3])!!
            Log.d(TAG, "Loading Part 4: ${part4File.name} (${part4File.length() / 1024 / 1024}MB)")
            interpreter = createInterpreter(part4File)

            val packedShape = interpreter.getOutputTensor(0).shape()
            val gaussianCount = packedShape[1]
            val packedSize = packedShape.fold(1) { acc, d -> acc * d }
            val packedBuf = ByteBuffer.allocateDirect(packedSize * 4).apply { order(ByteOrder.nativeOrder()) }

            inputBuffer.rewind()
            val part4Inputs = arrayOf<Any>(
                inputBuffer,
                loadTensorFromFile(File(tempDir, "image_tokens.tensor")),
                loadTensorFromFile(File(tempDir, "latent0.tensor")),
                loadTensorFromFile(File(tempDir, "latent1.tensor")),
                loadTensorFromFile(File(tempDir, "x0_feat.tensor")),
                loadTensorFromFile(File(tempDir, "x1_feat.tensor")),
                loadTensorFromFile(File(tempDir, "x2_feat.tensor")),
            )
            val part4OutMap = HashMap<Int, Any>()
            part4OutMap[0] = packedBuf

            val part4Start = System.currentTimeMillis()
            interpreter.runForMultipleInputsOutputs(part4Inputs, part4OutMap)
            val part4Time = System.currentTimeMillis() - part4Start
            Log.d(TAG, "Part 4 done in ${part4Time}ms. Gaussians: $gaussianCount")

            interpreter.close()
            System.gc()

            // Clean up temp files
            tempDir.listFiles()?.forEach { it.delete() }

            // Step 8: Parse and write PLY
            progressCallback?.invoke(0.75f, "Extracting $gaussianCount Gaussians...")
            packedBuf.rewind()
            val gaussianParams = FloatArray(gaussianCount * PARAMS_PER_GAUSSIAN)
            packedBuf.asFloatBuffer().get(gaussianParams)

            progressCallback?.invoke(0.80f, "Writing PLY ($gaussianCount Gaussians)...")
            val result = writeGaussianPly(gaussianParams, gaussianCount, progressCallback)

            val totalTime = System.currentTimeMillis() - startTime
            Log.d(TAG, "Single-Patch LiteRT completed: $gaussianCount Gaussians in ${totalTime}ms")
            Log.d(TAG, "  P1=${part1Time}ms P2=${part2Time}ms merge=${mergeTime}ms P3=${part3Time}ms P4=${part4Time}ms")

            progressCallback?.invoke(1.0f, "Done!")
            return result

        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OUT OF MEMORY during single-patch LiteRT inference", e)
            System.gc()
            tempDir.listFiles()?.forEach { it.delete() }
            progressCallback?.invoke(0f, "Out of memory")
            return null
        } catch (e: Exception) {
            Log.e(TAG, "Single-patch LiteRT inference failed", e)
            tempDir.listFiles()?.forEach { it.delete() }
            progressCallback?.invoke(0f, "Error: ${e.message}")
            return null
        }
    }

    /**
     * Save a ByteBuffer tensor to file with shape metadata.
     * Format: [numDims (4 bytes)] [dim0 (8 bytes)] [dim1 (8 bytes)] ... [float data]
     */
    private fun saveTensorToFile(buffer: ByteBuffer, shape: IntArray, file: File) {
        buffer.rewind()

        RandomAccessFile(file, "rw").use { raf ->
            val channel = raf.channel

            // Write shape metadata
            val headerSize = 4 + shape.size * 8
            val headerBuffer = ByteBuffer.allocate(headerSize)
            headerBuffer.order(ByteOrder.LITTLE_ENDIAN)
            headerBuffer.putInt(shape.size)
            shape.forEach { headerBuffer.putLong(it.toLong()) }
            headerBuffer.flip()
            channel.write(headerBuffer)

            // Write float data in chunks
            val totalBytes = buffer.remaining()
            val chunkSize = 4 * 1024 * 1024 // 4MB chunks
            val chunkBuffer = ByteBuffer.allocateDirect(chunkSize)
            chunkBuffer.order(ByteOrder.nativeOrder())

            var written = 0
            while (written < totalBytes) {
                val bytesToWrite = minOf(chunkSize, totalBytes - written)
                chunkBuffer.clear()
                chunkBuffer.limit(bytesToWrite)

                // Copy from source buffer
                val srcSlice = buffer.slice()
                srcSlice.order(ByteOrder.nativeOrder())
                srcSlice.limit(bytesToWrite)
                chunkBuffer.put(srcSlice)
                buffer.position(buffer.position() + bytesToWrite)

                chunkBuffer.flip()
                channel.write(chunkBuffer)
                written += bytesToWrite
            }
        }
    }

    /**
     * Load a tensor from file, returning a DirectByteBuffer with the float data.
     */
    private fun loadTensorFromFile(file: File): ByteBuffer {
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

            // Calculate data size
            val dataOffset = 4L + numDims * 8L
            val numElements = shape.fold(1L) { acc, dim -> acc * dim }
            val dataSize = numElements * 4

            // Memory-map the data portion
            val mappedBuffer = channel.map(FileChannel.MapMode.READ_ONLY, dataOffset, dataSize)
            mappedBuffer.order(ByteOrder.nativeOrder())

            Log.d(TAG, "Loaded tensor: shape=${shape.contentToString()}, size=${dataSize / 1024}KB")
            return mappedBuffer
        }
    }

    // ========================================================================
    // Single mode: full model in one pass (original behavior)
    // ========================================================================

    private suspend fun inferSingleMode(
        bitmap: Bitmap,
        progressCallback: ((Float, String) -> Unit)?
    ): StreamingResult? {
        try {
            val startTime = System.currentTimeMillis()

            // Step 1: Preprocess image
            progressCallback?.invoke(0.05f, "Preprocessing image...")
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
            val inputBuffer = preprocessImage(scaledBitmap)
            scaledBitmap.recycle()
            Log.d(TAG, "Image preprocessed to ${IMAGE_SIZE}x${IMAGE_SIZE}")

            // Step 2: Load interpreter
            progressCallback?.invoke(0.10f, "Loading SHARP model (LiteRT)...")
            val modelFile = findSingleModel()!!
            Log.d(TAG, "Loading model: ${modelFile.name} (${modelFile.length() / 1024 / 1024}MB)")
            val interpreter = createInterpreter(modelFile)

            val outputTensor = interpreter.getOutputTensor(0)
            val outputShape = outputTensor.shape()
            val gaussianCount = outputShape[1]
            Log.d(TAG, "Model loaded. Output shape: ${outputShape.contentToString()}")

            val outputSize = gaussianCount * PARAMS_PER_GAUSSIAN * 4
            val outputBuffer = ByteBuffer.allocateDirect(outputSize).apply {
                order(ByteOrder.nativeOrder())
            }

            // Step 3: Run inference
            progressCallback?.invoke(0.15f, "Running SHARP inference...")
            val inferenceStart = System.currentTimeMillis()
            interpreter.run(inputBuffer, outputBuffer)
            val inferenceTime = System.currentTimeMillis() - inferenceStart
            Log.d(TAG, "Inference completed in ${inferenceTime}ms ($gaussianCount Gaussians)")

            interpreter.close()
            System.gc()

            // Step 4: Parse output
            progressCallback?.invoke(0.60f, "Extracting $gaussianCount Gaussians...")
            outputBuffer.rewind()
            val gaussianParams = FloatArray(gaussianCount * PARAMS_PER_GAUSSIAN)
            outputBuffer.asFloatBuffer().get(gaussianParams)

            // Step 5: Write PLY
            progressCallback?.invoke(0.70f, "Writing PLY ($gaussianCount Gaussians)...")
            val result = writeGaussianPly(gaussianParams, gaussianCount, progressCallback)

            val totalTime = System.currentTimeMillis() - startTime
            Log.d(TAG, "Single LiteRT SHARP completed: $gaussianCount Gaussians in ${totalTime}ms")

            progressCallback?.invoke(1.0f, "Done!")
            return result

        } catch (e: Exception) {
            Log.e(TAG, "Single LiteRT inference failed", e)
            progressCallback?.invoke(0f, "Error: ${e.message}")
            return null
        }
    }

    // ========================================================================
    // Shared utilities
    // ========================================================================

    /**
     * Preprocess image to [1, 3, 1536, 1536] CHW format, normalized [0, 1].
     */
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val bufferSize = 1 * 3 * width * height * 4
        val buffer = ByteBuffer.allocateDirect(bufferSize).apply {
            order(ByteOrder.nativeOrder())
        }

        // CHW format: all R, then all G, then all B
        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255f)
        }
        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 8) and 0xFF) / 255f)
        }
        for (pixel in pixels) {
            buffer.putFloat((pixel and 0xFF) / 255f)
        }

        buffer.rewind()
        return buffer
    }

    /**
     * Write Gaussians to PLY file from interleaved [N, 14] format.
     */
    private fun writeGaussianPly(
        gaussianParams: FloatArray,
        gaussianCount: Int,
        progressCallback: ((Float, String) -> Unit)?
    ): StreamingResult {
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
            val channel = fos.channel

            val headerBytes = header.toByteArray(Charsets.UTF_8)
            val headerBuffer = ByteBuffer.allocateDirect(headerBytes.size)
            headerBuffer.put(headerBytes)
            headerBuffer.flip()
            channel.write(headerBuffer)

            val batchSize = 512
            val batchBuffer = ByteBuffer.allocateDirect(BYTES_PER_VERTEX * batchSize)
            batchBuffer.order(ByteOrder.LITTLE_ENDIAN)
            val scaleBoost = 1.3f
            val minScale = 0.001f
            val lutScale = (LOGIT_LUT_SIZE - 1).toFloat()

            val progressEvery = max(1, gaussianCount / 10)
            var processed = 0

            while (processed < gaussianCount) {
                val currentBatch = minOf(batchSize, gaussianCount - processed)

                for (j in 0 until currentBatch) {
                    val offset = (processed + j) * PARAMS_PER_GAUSSIAN

                    // Position (flip Y and Z for coordinate system)
                    val posX = gaussianParams[offset + 0]
                    val posY = -gaussianParams[offset + 1]
                    val posZ = -gaussianParams[offset + 2]

                    if (posX < minX) minX = posX; if (posX > maxX) maxX = posX
                    if (posY < minY) minY = posY; if (posY > maxY) maxY = posY
                    if (posZ < minZ) minZ = posZ; if (posZ > maxZ) maxZ = posZ

                    batchBuffer.putFloat(posX)
                    batchBuffer.putFloat(posY)
                    batchBuffer.putFloat(posZ)

                    // Normals (unused)
                    batchBuffer.putFloat(0f)
                    batchBuffer.putFloat(0f)
                    batchBuffer.putFloat(0f)

                    // Color (offset 11-13) → SH DC coefficients
                    val colorR = gaussianParams[offset + 11].coerceIn(0f, 1f)
                    val colorG = gaussianParams[offset + 12].coerceIn(0f, 1f)
                    val colorB = gaussianParams[offset + 13].coerceIn(0f, 1f)
                    batchBuffer.putFloat((colorR - 0.5f) / SH_C0)
                    batchBuffer.putFloat((colorG - 0.5f) / SH_C0)
                    batchBuffer.putFloat((colorB - 0.5f) / SH_C0)

                    // Higher order SH (45 zeros)
                    repeat(45) { batchBuffer.putFloat(0f) }

                    // Opacity (offset 3, raw logit → sigmoid → logit for PLY)
                    val rawOpacityLogit = gaussianParams[offset + 3]
                    val opacitySigmoid = 1f / (1f + kotlin.math.exp(-rawOpacityLogit))
                    val opacityLutIndex = (opacitySigmoid * lutScale).toInt()
                        .coerceIn(0, LOGIT_LUT_SIZE - 1)
                    batchBuffer.putFloat(LOGIT_LUT[opacityLutIndex])

                    // Scale (offset 4-6) → log transform
                    batchBuffer.putFloat(lnLut(max(gaussianParams[offset + 4] * scaleBoost, minScale)))
                    batchBuffer.putFloat(lnLut(max(gaussianParams[offset + 5] * scaleBoost, minScale)))
                    batchBuffer.putFloat(lnLut(max(gaussianParams[offset + 6] * scaleBoost, minScale)))

                    // Rotation quaternion (offset 7-10) → normalize
                    val quatW = gaussianParams[offset + 7]
                    val quatX = gaussianParams[offset + 8]
                    val quatY = gaussianParams[offset + 9]
                    val quatZ = gaussianParams[offset + 10]
                    val quatMag = sqrt(quatW * quatW + quatX * quatX + quatY * quatY + quatZ * quatZ)
                    val quatInvMag = if (quatMag > 1e-8f) 1f / quatMag else 1f
                    batchBuffer.putFloat(quatW * quatInvMag)
                    batchBuffer.putFloat(quatX * quatInvMag)
                    batchBuffer.putFloat(quatY * quatInvMag)
                    batchBuffer.putFloat(quatZ * quatInvMag)
                }

                batchBuffer.flip()
                batchBuffer.limit(currentBatch * BYTES_PER_VERTEX)
                while (batchBuffer.hasRemaining()) {
                    channel.write(batchBuffer)
                }
                batchBuffer.clear()

                processed += currentBatch
                if (processed % progressEvery == 0 || processed == gaussianCount) {
                    val progress = 0.70f + (processed.toFloat() / gaussianCount) * 0.25f
                    progressCallback?.invoke(progress, "Writing PLY ($processed/$gaussianCount)...")
                }
            }
        }

        plyFile.copyTo(classicPlyFile, overwrite = true)

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
        isInitialized = false
        tempDir.listFiles()?.forEach { it.delete() }
        Log.d(TAG, "LiteRTSharp released")
    }
}
