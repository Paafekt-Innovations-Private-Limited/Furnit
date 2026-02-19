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
import kotlin.text.Regex

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

        // Pre-computed sigmoid LUT: avoids Math.exp() per Gaussian in PLY writing.
        // Maps logit input range [-12, 12] to sigmoid(x).  BLAS-equivalent: vectorized
        // activation function applied via table lookup (like vvrecf + vvexpf combo).
        private const val SIGMOID_LUT_SIZE = 2048
        private const val SIGMOID_LUT_MIN = -12f
        private const val SIGMOID_LUT_MAX = 12f
        private val SIGMOID_LUT_SCALE = (SIGMOID_LUT_SIZE - 1).toFloat() / (SIGMOID_LUT_MAX - SIGMOID_LUT_MIN)
        private val SIGMOID_LUT = FloatArray(SIGMOID_LUT_SIZE) { i ->
            val x = SIGMOID_LUT_MIN + (SIGMOID_LUT_MAX - SIGMOID_LUT_MIN) * i / (SIGMOID_LUT_SIZE - 1)
            (1.0 / (1.0 + kotlin.math.exp(-x.toDouble()))).toFloat()
        }

        private fun sigmoidLut(x: Float): Float {
            if (x <= SIGMOID_LUT_MIN) return 0f
            if (x >= SIGMOID_LUT_MAX) return 1f
            return SIGMOID_LUT[((x - SIGMOID_LUT_MIN) * SIGMOID_LUT_SCALE).toInt()]
        }

        // Pre-allocated 45-zero block for SH rest coefficients (180 bytes).
        // Written once per batch with bulk put instead of 45 individual putFloat(0f) calls.
        private val ZERO_SH_BLOCK = ByteArray(45 * 4)

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

    // Reusable 4MB DirectByteBuffer for saveTensorToFile. Avoids creating 70+ transient
    // DirectByteBuffers during patch loops. DirectByteBuffers are freed by GC finalizers
    // which may not run promptly, causing native memory pressure and LMK kills.
    private val reusableSaveChunk: ByteBuffer by lazy {
        ByteBuffer.allocateDirect(4 * 1024 * 1024).apply { order(ByteOrder.nativeOrder()) }
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

        val modelSizeMB = modelFile.length() / 1024 / 1024
        Log.d(TAG, "Single mode: ${modelFile.absolutePath} (${modelSizeMB}MB)")

        // Safety: the single LiteRT model can be >1GB and frequently causes the app to be
        // killed by low-memory killer on real devices. Prefer split files instead.
        // (Your log shows vit_gaussian_fp16.tflite ~1258MB.)
        val maxSingleModelMB = 700L
        if (modelSizeMB > maxSingleModelMB) {
            Log.e(TAG, "Single LiteRT model too large (${modelSizeMB}MB). Push split models instead: ${SPLIT_PART_FILENAMES.joinToString()}")
            return false
        }
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
    private fun createInterpreter(
        modelFile: File,
        numThreadsOverride: Int? = null,
        useXnnpack: Boolean = true
    ): Interpreter {
        val modelSizeMB = modelFile.length() / 1024 / 1024
        val numThreads = numThreadsOverride ?: Runtime.getRuntime().availableProcessors()
        val useNnapi = BackendConfig.ENABLE_LITERT_NNAPI

        // Try GPU delegate for small-enough models on supported devices.
        // Disabled by default due to device-specific native crashes.
        if (BackendConfig.ENABLE_LITERT_GPU && modelSizeMB < 500) {
            var gpuDelegate: GpuDelegate? = null
            try {
                val compatList = CompatibilityList()
                if (compatList.isDelegateSupportedOnThisDevice) {
                    val gpuOptions = compatList.bestOptionsForThisDevice
                    gpuDelegate = GpuDelegate(gpuOptions)
                    val gpuInterpreterOptions = Interpreter.Options().apply {
                        setNumThreads(numThreads)
                        if (useNnapi) {
                            // Allow partial NNAPI delegation for ops not covered by GPU delegate.
                            try { setUseNNAPI(true) } catch (_: Throwable) {}
                        }
                        addDelegate(gpuDelegate)
                    }
                    Log.d(TAG, "Creating GPU interpreter for ${modelFile.name} (threads=$numThreads)...")
                    val interpreter = Interpreter(modelFile, gpuInterpreterOptions)
                    Log.d(TAG, "GPU interpreter created for ${modelFile.name}")
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

        // CPU path — try NNAPI first (hardware acceleration), fall back to pure CPU+XNNPACK
        if (useNnapi) {
            try {
                val nnapiOptions = Interpreter.Options().apply {
                    setNumThreads(numThreads)
                    try { setUseXNNPACK(useXnnpack) } catch (_: Throwable) {}
                    setUseNNAPI(true)
                    setAllowFp16PrecisionForFp32(true)
                }
                Log.d(TAG, "Creating NNAPI interpreter for ${modelFile.name} (${modelSizeMB}MB, threads=$numThreads)...")
                val interpreter = Interpreter(modelFile, nnapiOptions)
                Log.d(TAG, "NNAPI interpreter created for ${modelFile.name}")
                return interpreter
            } catch (e: Throwable) {
                Log.w(TAG, "NNAPI failed for ${modelFile.name}: ${e.message} — falling back to CPU+XNNPACK")
            }
        }

        // Pure CPU + XNNPACK (no NNAPI)
        val cpuOptions = Interpreter.Options().apply {
            setNumThreads(numThreads)
            try { setUseXNNPACK(useXnnpack) } catch (_: Throwable) {}
            setAllowFp16PrecisionForFp32(true)
        }
        Log.d(
            TAG,
            "Creating CPU interpreter for ${modelFile.name} (${modelSizeMB}MB, threads=$numThreads, xnnpack=$useXnnpack)..."
        )
        val interpreter = Interpreter(modelFile, cpuOptions)
        Log.d(TAG, "CPU interpreter created for ${modelFile.name}")
        return interpreter
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
     *
     * Optimized: pre-compute row/col indices and weights to avoid redundant
     * float math in the inner loop. Similar to iOS MPSImageBilinearScale approach.
     */
    private fun resizeImageBuffer(src: ByteBuffer, srcSize: Int, dstSize: Int): ByteBuffer {
        src.rewind()
        val srcFloats = FloatArray(3 * srcSize * srcSize)
        src.asFloatBuffer().get(srcFloats)

        val scale = srcSize.toFloat() / dstSize
        val srcSizeM2 = srcSize - 2

        // Pre-compute x indices/weights once (reused per row)
        val xIndices = IntArray(dstSize)
        val xWeights = FloatArray(dstSize)
        for (dx in 0 until dstSize) {
            val sx = dx * scale
            xIndices[dx] = sx.toInt().coerceIn(0, srcSizeM2)
            xWeights[dx] = sx - xIndices[dx]
        }

        // Write directly to output buffer; row buffer avoids full dstFloats allocation (~7MB)
        val buf = ByteBuffer.allocateDirect(3 * dstSize * dstSize * 4).apply { order(ByteOrder.nativeOrder()) }
        val dstFb = buf.asFloatBuffer()
        val rowBuf = FloatArray(dstSize)

        for (c in 0..2) {
            val srcChanOff = c * srcSize * srcSize
            val dstChanOff = c * dstSize * dstSize
            for (dy in 0 until dstSize) {
                val sy0 = (dy * scale).toInt().coerceIn(0, srcSizeM2)
                val fy = dy * scale - sy0
                val oneMinusFy = 1f - fy
                val srcRow0 = srcChanOff + sy0 * srcSize
                val srcRow1 = srcRow0 + srcSize
                for (dx in 0 until dstSize) {
                    val sx0 = xIndices[dx]
                    val fx = xWeights[dx]
                    rowBuf[dx] =
                        oneMinusFy * (srcFloats[srcRow0 + sx0] + fx * (srcFloats[srcRow0 + sx0 + 1] - srcFloats[srcRow0 + sx0])) +
                        fy * (srcFloats[srcRow1 + sx0] + fx * (srcFloats[srcRow1 + sx0 + 1] - srcFloats[srcRow1 + sx0]))
                }
                dstFb.position(dstChanOff + dy * dstSize)
                dstFb.put(rowBuf)
            }
        }
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
     * Reshape tokens [1, 577, 1024] → spatial [1024, 24, 24] directly from file
     * into a pre-allocated FloatArray. Avoids intermediate heap allocation.
     *
     * BLAS-like optimization: single bulk FloatBuffer.get() (~590K floats) instead of
     * ~590K individual fb.get(idx) calls — each individual call crosses JNI boundary.
     * Bulk get() uses a single memcpy internally (like cblas_scopy).
     */
    private fun reshapeTokensToSpatialDirect(tokensFile: File, output: FloatArray) {
        val buf = loadTensorFromFile(tokensFile)
        val fb = buf.asFloatBuffer()
        // One bulk read: 577 * 1024 floats in a single JNI call
        val allTokens = FloatArray(577 * FEATURE_DIM)
        fb.get(allTokens)

        // Transpose from [577, 1024] (seq, channel) → [1024, 24, 24] (channel, h, w)
        // Skip CLS token (index 0), use positions 1..576
        val gridArea = GRID_H * GRID_W
        for (h in 0 until GRID_H) {
            for (w in 0 until GRID_W) {
                val seqIdx = NUM_PREFIX_TOKENS + h * GRID_W + w
                val srcBase = seqIdx * FEATURE_DIM
                val dstSpatialIdx = h * GRID_W + w
                for (c in 0 until FEATURE_DIM) {
                    output[c * gridArea + dstSpatialIdx] = allTokens[srcBase + c]
                }
            }
        }
        // allTokens + buf eligible for GC immediately
    }

    /**
     * Merge a single patch's spatial data into the merged output array in-place.
     * Used for streaming merge to avoid materializing all patches simultaneously.
     *
     * BLAS-like optimization: uses System.arraycopy for contiguous row segments
     * instead of element-by-element assignment. System.arraycopy is a JVM intrinsic
     * that maps to optimized memcpy/SIMD on ARM (like cblas_scopy for contiguous blocks).
     * Reduces ~331K individual assignments to ~18K arraycopy calls per patch.
     */
    private fun mergeSinglePatchInPlace(
        merged: FloatArray,
        patch: FloatArray,
        gridJ: Int,
        gridI: Int,
        gridSteps: Int,
        padding: Int,
        outH: Int,
        outW: Int
    ) {
        val patchSpatialSize = GRID_H
        val patchArea = patchSpatialSize * patchSpatialSize
        val outArea = outH * outW
        val rowStart = if (gridJ != 0) padding else 0
        val rowEnd = if (gridJ != gridSteps - 1) patchSpatialSize - padding else patchSpatialSize
        val colStart = if (gridI != 0) padding else 0
        val colEnd = if (gridI != gridSteps - 1) patchSpatialSize - padding else patchSpatialSize
        val copyLength = colEnd - colStart
        val dstRowOff = if (gridJ == 0) 0 else padding + gridJ * (patchSpatialSize - 2 * padding)
        val dstColOff = if (gridI == 0) 0 else padding + gridI * (patchSpatialSize - 2 * padding)

        for (c in 0 until FEATURE_DIM) {
            val srcChanBase = c * patchArea
            val dstChanBase = c * outArea
            for (r in rowStart until rowEnd) {
                val dstRow = dstRowOff + (r - rowStart)
                // Contiguous row copy: patch[srcChanBase + r*24 + colStart .. colEnd]
                // → merged[dstChanBase + dstRow*outW + dstColOff .. +copyLength]
                System.arraycopy(
                    patch, srcChanBase + r * patchSpatialSize + colStart,
                    merged, dstChanBase + dstRow * outW + dstColOff,
                    copyLength
                )
            }
        }
    }

    /**
     * Streaming merge: loads one patch at a time, reshapes, and merges in-place.
     * Peak memory: 1 patch spatial (~2.4MB) + merged output (~38MB),
     * vs. old approach: N patches at once (~60MB) + merged output.
     * This mirrors iOS's Accelerate-style sequential processing.
     */
    private fun streamingMerge(
        filePrefix: String,
        startIndex: Int,
        gridSteps: Int,
        padding: Int,
        isTokenReshape: Boolean
    ): ByteBuffer {
        val patchSpatialSize = GRID_H
        val outH = patchSpatialSize + (gridSteps - 1) * (patchSpatialSize - 2 * padding)
        val outW = outH

        val merged = FloatArray(FEATURE_DIM * outH * outW)
        val patchSpatial = FloatArray(FEATURE_DIM * GRID_H * GRID_W)  // reused across all patches

        var idx = 0
        for (j in 0 until gridSteps) {
            for (i in 0 until gridSteps) {
                val fileIndex = startIndex + idx
                val file = File(tempDir, "${filePrefix}_$fileIndex.tensor")

                if (isTokenReshape) {
                    reshapeTokensToSpatialDirect(file, patchSpatial)
                } else {
                    val buf = loadTensorFromFile(file)
                    buf.asFloatBuffer().get(patchSpatial, 0, patchSpatial.size)
                }

                mergeSinglePatchInPlace(merged, patchSpatial, j, i, gridSteps, padding, outH, outW)
                idx++
            }
        }

        val buf = ByteBuffer.allocateDirect(merged.size * 4).apply { order(ByteOrder.nativeOrder()) }
        buf.asFloatBuffer().put(merged)
        buf.rewind()
        return buf
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
            var inputBuffer: ByteBuffer? = preprocessImage(scaledBitmap)
            scaledBitmap.recycle()
            Log.d(TAG, "Image preprocessed to ${IMAGE_SIZE}x${IMAGE_SIZE}")

            // Step 2: Create pyramid and extract patches
            progressCallback?.invoke(0.03f, "Creating pyramid patches...")
            var x1Buffer: ByteBuffer? = resizeImageBuffer(inputBuffer!!, IMAGE_SIZE, IMAGE_SIZE / 2)  // 768
            var x2Buffer: ByteBuffer? = resizeImageBuffer(inputBuffer!!, IMAGE_SIZE, IMAGE_SIZE / 4)  // 384

            // IMPORTANT: Don't materialize all 35 patch buffers at once.
            // Keeping 35 direct ByteBuffers in memory can trigger OOM around the
            // first interpreter load/run on memory-constrained devices.
            val totalPatches = NUM_PATCHES
            val patchFloatCount = 3 * PATCH_SIZE * PATCH_SIZE
            val tempRow = FloatArray(PATCH_SIZE)

            fun extractPatchFromBuffer(
                srcBuffer: ByteBuffer,
                srcSize: Int,
                i0: Int,
                j0: Int
            ): ByteBuffer {
                val channelSize = srcSize * srcSize
                val src = srcBuffer.asFloatBuffer()
                val patchBuf = ByteBuffer.allocateDirect(patchFloatCount * 4).apply {
                    order(ByteOrder.nativeOrder())
                }
                val dst = patchBuf.asFloatBuffer()
                for (c in 0..2) {
                    val chanOffset = c * channelSize
                    for (row in 0 until PATCH_SIZE) {
                        val srcIdx = chanOffset + (j0 + row) * srcSize + i0
                        src.position(srcIdx)
                        src.get(tempRow, 0, PATCH_SIZE)
                        dst.put(tempRow, 0, PATCH_SIZE)
                    }
                }
                patchBuf.rewind()
                return patchBuf
            }

            // Compute patch grids for x0 (5x5, overlap 0.25) and x1 (3x3, overlap 0.5)
            val x0Stride = (PATCH_SIZE * (1f - 0.25f)).toInt()   // 288
            val x0Steps = ((IMAGE_SIZE - PATCH_SIZE).toFloat() / x0Stride).toInt() + 1
            val x1Size = IMAGE_SIZE / 2
            val x1Stride = (PATCH_SIZE * (1f - 0.5f)).toInt()    // 192
            val x1Steps = ((x1Size - PATCH_SIZE).toFloat() / x1Stride).toInt() + 1

            Log.d(TAG, "Patch grid: x0Steps=$x0Steps (expected 5), x1Steps=$x1Steps (expected 3), total=$totalPatches")

            // Step 3: Load Part 1, run on each patch
            progressCallback?.invoke(0.05f, "Part 1: Encoding patches (0/$totalPatches)...")
            val part1File = findFile(SPLIT_PART_FILENAMES[0])!!
            Log.d(TAG, "Loading Part 1: ${part1File.name} (${part1File.length() / 1024 / 1024}MB)")
            var interpreter = createInterpreter(part1File, numThreadsOverride = 4, useXnnpack = true)

            val tokensShape = interpreter.getOutputTensor(0).shape()
            val tokensSize = tokensShape.fold(1) { acc, d -> acc * d }
            val block5Shape = interpreter.getOutputTensor(1).shape()
            val block5Size = block5Shape.fold(1) { acc, d -> acc * d }

            // Pre-allocate reusable output buffers — avoids 35×2 DirectByteBuffer allocations
            // that would linger until GC. Mirrors iOS zero-copy buffer approach.
            val reusableTokensBuf = ByteBuffer.allocateDirect(tokensSize * 4).apply { order(ByteOrder.nativeOrder()) }
            val reusableBlock5Buf = ByteBuffer.allocateDirect(block5Size * 4).apply { order(ByteOrder.nativeOrder()) }
            val reusableOutputMap = HashMap<Int, Any>(2)

            val part1Start = System.currentTimeMillis()
            var patchIndex = 0

            // Helper to run one patch through Part 1 and save results
            fun runPart1Patch(patchBuf: ByteBuffer) {
                reusableTokensBuf.clear()
                reusableBlock5Buf.clear()
                reusableOutputMap.clear()
                reusableOutputMap[0] = reusableTokensBuf
                reusableOutputMap[1] = reusableBlock5Buf
                interpreter.runForMultipleInputsOutputs(arrayOf(patchBuf), reusableOutputMap)
                saveTensorToFile(reusableTokensBuf, tokensShape, File(tempDir, "tokens_$patchIndex.tensor"))
                saveTensorToFile(reusableBlock5Buf, block5Shape, File(tempDir, "block5_$patchIndex.tensor"))
                patchIndex++
            }

            // x0 patches (5x5)
            for (j in 0 until x0Steps) {
                val j0 = j * x0Stride
                for (i in 0 until x0Steps) {
                    val i0 = i * x0Stride
                    if (patchIndex % 5 == 0) {
                        val pct = 0.05f + (patchIndex.toFloat() / totalPatches) * 0.25f
                        progressCallback?.invoke(pct, "Part 1: Encoding patches ($patchIndex/$totalPatches)...")
                    }
                    val patchBuf = extractPatchFromBuffer(inputBuffer, IMAGE_SIZE, i0, j0)
                    runPart1Patch(patchBuf)
                }
            }

            // x1 patches (3x3)
            for (j in 0 until x1Steps) {
                val j0 = j * x1Stride
                for (i in 0 until x1Steps) {
                    val i0 = i * x1Stride
                    if (patchIndex % 5 == 0) {
                        val pct = 0.05f + (patchIndex.toFloat() / totalPatches) * 0.25f
                        progressCallback?.invoke(pct, "Part 1: Encoding patches ($patchIndex/$totalPatches)...")
                    }
                    val patchBuf = extractPatchFromBuffer(x1Buffer!!, x1Size, i0, j0)
                    runPart1Patch(patchBuf)
                }
            }

            // Release x1Buffer (~7MB DirectByteBuffer) — no longer needed after x1 patches
            x1Buffer = null

            // x2 patch (single 384x384 already prepared)
            run {
                val pct = 0.05f + (patchIndex.toFloat() / totalPatches) * 0.25f
                progressCallback?.invoke(pct, "Part 1: Encoding patches ($patchIndex/$totalPatches)...")
            }
            x2Buffer!!.rewind()
            runPart1Patch(x2Buffer!!)

            // Release x2Buffer (~1.8MB DirectByteBuffer) — no longer needed
            x2Buffer = null

            if (patchIndex != totalPatches) {
                Log.w(TAG, "Patch count mismatch: processed=$patchIndex expected=$totalPatches")
            }
            interpreter.close()
            // Pyramid buffers released above; GC can now reclaim ~9MB of direct memory
            System.gc()
            val part1Time = System.currentTimeMillis() - part1Start
            Log.d(TAG, "Part 1 done: $totalPatches patches in ${part1Time}ms (${part1Time / totalPatches}ms/patch)")

            // Step 4: Load Part 2, run on each token set
            progressCallback?.invoke(0.30f, "Part 2: Processing features (0/$totalPatches)...")
            val part2File = findFile(SPLIT_PART_FILENAMES[1])!!
            Log.d(TAG, "Loading Part 2: ${part2File.name} (${part2File.length() / 1024 / 1024}MB)")
            interpreter = createInterpreter(part2File, numThreadsOverride = 4, useXnnpack = true)

            val featShape = interpreter.getOutputTensor(0).shape()
            val featSize = featShape.fold(1) { acc, d -> acc * d }

            // Reusable output buffer for Part 2 — same zero-alloc principle as Part 1
            val reusableFeatBuf = ByteBuffer.allocateDirect(featSize * 4).apply { order(ByteOrder.nativeOrder()) }
            val part2OutputMap = HashMap<Int, Any>(1)

            val part2Start = System.currentTimeMillis()
            for (i in 0 until totalPatches) {
                if (i % 5 == 0) {
                    val pct = 0.30f + (i.toFloat() / totalPatches) * 0.15f
                    progressCallback?.invoke(pct, "Part 2: Processing features ($i/$totalPatches)...")
                }

                val tokensInput = loadTensorFromFile(File(tempDir, "tokens_$i.tensor"))
                reusableFeatBuf.clear()
                part2OutputMap.clear()
                part2OutputMap[0] = reusableFeatBuf
                interpreter.runForMultipleInputsOutputs(arrayOf(tokensInput), part2OutputMap)

                saveTensorToFile(reusableFeatBuf, featShape, File(tempDir, "feat_$i.tensor"))
            }
            interpreter.close()
            System.gc()
            val part2Time = System.currentTimeMillis() - part2Start
            Log.d(TAG, "Part 2 done: $totalPatches patches in ${part2Time}ms (${part2Time / totalPatches}ms/patch)")

            // Step 5: Streaming merge — process ONE patch at a time (iOS Accelerate-style)
            // Old approach materialized all 25 patches (~60MB) in lists; this streams through
            // with a single reusable 2.4MB buffer, reducing peak merge memory from ~250MB to ~40MB.
            progressCallback?.invoke(0.45f, "Merging patch features...")
            val mergeStart = System.currentTimeMillis()
            val latent0OutH = GRID_H + 4 * (GRID_H - 2 * X0_MERGE_PADDING)  // 24 + 4*18 = 96
            val x1OutH = GRID_H + 2 * (GRID_H - 2 * X1_MERGE_PADDING)  // 24 + 2*12 = 48

            // Each merge produces a ~38MB DirectByteBuffer. Save to disk and release
            // immediately to avoid holding 4×38MB = 152MB of direct memory simultaneously.
            // This mimics iOS's pattern of releasing buffers as soon as they're persisted.

            // latent0: block5 tokens reshaped (x0 patches, 5x5 grid)
            Log.d(TAG, "Merge: streaming latent0 (block5, 25 patches)...")
            streamingMerge("block5", 0, 5, X0_MERGE_PADDING, isTokenReshape = true).let { buf ->
                saveTensorToFile(buf, intArrayOf(1, FEATURE_DIM, latent0OutH, latent0OutH),
                    File(tempDir, "latent0.tensor"))
            } // buf released here — ~38MB DirectByteBuffer eligible for GC

            // latent1: block11 tokens reshaped (x0 patches, 5x5 grid)
            Log.d(TAG, "Merge: streaming latent1 (tokens, 25 patches)...")
            streamingMerge("tokens", 0, 5, X0_MERGE_PADDING, isTokenReshape = true).let { buf ->
                saveTensorToFile(buf, intArrayOf(1, FEATURE_DIM, latent0OutH, latent0OutH),
                    File(tempDir, "latent1.tensor"))
            }

            // x0_feat: Part 2 features (x0 patches, 5x5 grid)
            Log.d(TAG, "Merge: streaming x0_feat (feat, 25 patches)...")
            streamingMerge("feat", 0, 5, X0_MERGE_PADDING, isTokenReshape = false).let { buf ->
                saveTensorToFile(buf, intArrayOf(1, FEATURE_DIM, latent0OutH, latent0OutH),
                    File(tempDir, "x0_feat.tensor"))
            }

            // x1_feat: Part 2 features (x1 patches, 3x3 grid)
            Log.d(TAG, "Merge: streaming x1_feat (feat, 9 patches)...")
            streamingMerge("feat", X0_PATCH_COUNT, 3, X1_MERGE_PADDING, isTokenReshape = false).let { buf ->
                saveTensorToFile(buf, intArrayOf(1, FEATURE_DIM, x1OutH, x1OutH),
                    File(tempDir, "x1_feat.tensor"))
            }

            // x2_feat: patch 34, no merging needed — just rename
            File(tempDir, "feat_34.tensor").copyTo(File(tempDir, "x2_feat.tensor"), overwrite = true)

            // Clean up individual patch files to free disk space before Part 4
            for (i in 0 until NUM_PATCHES) {
                File(tempDir, "tokens_$i.tensor").delete()
                File(tempDir, "block5_$i.tensor").delete()
                File(tempDir, "feat_$i.tensor").delete()
            }

            // Aggressive GC — merge DirectByteBuffers are now out of scope
            System.gc()
            Thread.sleep(100)
            System.gc()

            val mergeTime = System.currentTimeMillis() - mergeStart
            Log.d(TAG, "Streaming merge done in ${mergeTime}ms. Memory: ${getMemoryInfo()}")

            // Step 6: Part 3 — Image Encoder A (full image → image_tokens)
            progressCallback?.invoke(0.50f, "Part 3: Image encoder...")
            Log.d(TAG, "Memory before Part 3: ${getMemoryInfo()}")
            val part3File = findFile(SPLIT_PART_FILENAMES[2])!!
            Log.d(TAG, "Loading Part 3: ${part3File.name}")
            interpreter = createInterpreter(part3File, numThreadsOverride = 4, useXnnpack = true)
            val imgTokensShape = interpreter.getOutputTensor(0).shape()
            val imgTokensSize = imgTokensShape.fold(1) { acc, d -> acc * d }
            val imgTokensBuf = ByteBuffer.allocateDirect(imgTokensSize * 4).apply { order(ByteOrder.nativeOrder()) }

            inputBuffer!!.rewind()
            val part3Start = System.currentTimeMillis()
            interpreter.run(inputBuffer, imgTokensBuf)
            val part3Time = System.currentTimeMillis() - part3Start
            Log.d(TAG, "Part 3 done in ${part3Time}ms")

            saveTensorToFile(imgTokensBuf, imgTokensShape, File(tempDir, "image_tokens.tensor"))
            interpreter.close()

            // Save inputBuffer to disk so we can release the 27MB DirectByteBuffer BEFORE
            // creating the Part 4 interpreter. This mirrors iOS's strategy of releasing
            // intermediate buffers before heavy allocations.
            val inputImageFile = File(tempDir, "input_image.tensor")
            saveTensorToFile(inputBuffer!!, intArrayOf(1, 3, IMAGE_SIZE, IMAGE_SIZE), inputImageFile)
            inputBuffer = null  // Release 27MB DirectByteBuffer
            Log.d(TAG, "inputBuffer saved to disk and released (27MB freed)")

            System.gc()

            // Step 7: Part 4 — Image Encoder B + Full Decoder + Gaussians
            // CRITICAL: Part 4 model (386MB) causes large memory spike during
            // graph compilation. We MUST:
            //   1. Release ALL unnecessary buffers (inputBuffer, merge buffers) — done above
            //   2. Create interpreter with XNNPACK OFF (avoid the compile-time spike)
            //   3. THEN load input tensors from disk
            // This avoids the LMK kill that was happening at interpreter creation time.
            progressCallback?.invoke(0.60f, "Part 4: Loading decoder...")

            // Aggressive GC: reclaim DirectByteBuffers from merge phase + inputBuffer
            System.gc()
            Thread.sleep(200)
            System.gc()
            Log.d(TAG, "Memory before Part 4 interpreter: ${getMemoryInfo()}")

            val part4File = findFile(SPLIT_PART_FILENAMES[3])!!
            Log.d(TAG, "Loading Part 4: ${part4File.name} (${part4File.length() / 1024 / 1024}MB)")

            // Use 2 threads + XNNPACK OFF for Part 4 interpreter creation.
            // XNNPACK graph compilation for a 386MB model allocates ~800MB working memory,
            // which causes LMK kill. Without XNNPACK, TFLite uses its built-in CPU kernels
            // which have a much smaller compilation footprint. Inference is ~20% slower but
            // actually completes instead of being killed.
            interpreter = createInterpreter(part4File, numThreadsOverride = 2, useXnnpack = false)
            Log.d(TAG, "Part 4 interpreter created. Memory: ${getMemoryInfo()}")

            val packedShape = interpreter.getOutputTensor(0).shape()
            val gaussianCount = packedShape[1]
            val packedSize = packedShape.fold(1) { acc, d -> acc * d }
            val packedBuf = ByteBuffer.allocateDirect(packedSize * 4).apply { order(ByteOrder.nativeOrder()) }

            // NOW load Part 4 inputs — interpreter is stable, load tensors from disk.
            // Delete tensor files immediately after loading to free page cache.
            progressCallback?.invoke(0.62f, "Part 4: Loading tensors...")

            // Reload inputBuffer from disk (was saved before Part 4 interpreter creation)
            val inputImageForPart4 = loadTensorFromFile(inputImageFile)
            inputImageFile.delete()

            val imageTokensFile = File(tempDir, "image_tokens.tensor")
            val imageTokensTensorBuf = loadTensorFromFile(imageTokensFile)
            imageTokensFile.delete()

            val latent0File = File(tempDir, "latent0.tensor")
            val latent0TensorBuf = loadTensorFromFile(latent0File)
            latent0File.delete()

            val latent1File = File(tempDir, "latent1.tensor")
            val latent1TensorBuf = loadTensorFromFile(latent1File)
            latent1File.delete()

            val x0FeatFile = File(tempDir, "x0_feat.tensor")
            val x0FeatTensorBuf = loadTensorFromFile(x0FeatFile)
            x0FeatFile.delete()

            val x1FeatFile = File(tempDir, "x1_feat.tensor")
            val x1FeatTensorBuf = loadTensorFromFile(x1FeatFile)
            x1FeatFile.delete()

            val x2FeatFile = File(tempDir, "x2_feat.tensor")
            val x2FeatTensorBuf = loadTensorFromFile(x2FeatFile)
            x2FeatFile.delete()

            Log.d(TAG, "Part 4 inputs loaded. Memory: ${getMemoryInfo()}")

            val part4Inputs = buildPart4Inputs(
                interpreter = interpreter,
                image = inputImageForPart4,
                imageTokens = imageTokensTensorBuf,
                latent0 = latent0TensorBuf,
                latent1 = latent1TensorBuf,
                x0Feat = x0FeatTensorBuf,
                x1Feat = x1FeatTensorBuf,
                x2Feat = x2FeatTensorBuf
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
            progressCallback?.invoke(0.80f, "Writing PLY ($gaussianCount Gaussians)...")
            val result = writeGaussianPlyFromPackedBuffer(packedBuf, gaussianCount, progressCallback)

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
     *
     * Uses reusable 4MB DirectByteBuffer to avoid allocating transient direct buffers
     * on every call. Called 70+ times during patch loops — without reuse, each call
     * would allocate a 4MB DirectByteBuffer that persists until GC finalizer runs.
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

            // Write float data in chunks using reusable buffer
            val totalBytes = buffer.remaining()
            val chunkSize = reusableSaveChunk.capacity()

            var written = 0
            while (written < totalBytes) {
                val bytesToWrite = minOf(chunkSize, totalBytes - written)
                reusableSaveChunk.clear()
                reusableSaveChunk.limit(bytesToWrite)

                // Copy from source buffer
                val srcSlice = buffer.slice()
                srcSlice.order(ByteOrder.nativeOrder())
                srcSlice.limit(bytesToWrite)
                reusableSaveChunk.put(srcSlice)
                buffer.position(buffer.position() + bytesToWrite)

                reusableSaveChunk.flip()
                channel.write(reusableSaveChunk)
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
            val dataSizeBytes = numElements * 4L
            if (dataSizeBytes > Int.MAX_VALUE) {
                throw IllegalArgumentException("Tensor too large to load into a single ByteBuffer: ${dataSizeBytes} bytes (${file.name})")
            }

            // IMPORTANT: Avoid mmap here. MappedByteBuffers can keep native mappings alive
            // until GC runs a cleaner, which increases memory pressure and can trigger LMK.
            // Explicitly reading keeps lifetime bounded to the returned buffer.
            val dataBuffer = ByteBuffer.allocateDirect(dataSizeBytes.toInt()).apply {
                order(ByteOrder.nativeOrder())
            }
            channel.position(dataOffset)
            while (dataBuffer.hasRemaining()) {
                val read = channel.read(dataBuffer)
                if (read < 0) break
            }
            dataBuffer.flip()

            Log.d(TAG, "Loaded tensor: shape=${shape.contentToString()}, size=${dataSizeBytes / 1024}KB")
            return dataBuffer
        }
    }

    private fun parseServingArgIndex(tensorName: String?): Int? {
        if (tensorName.isNullOrBlank()) return null
        val match = Regex("args_(\\d+)").find(tensorName) ?: return null
        return match.groupValues.getOrNull(1)?.toIntOrNull()
    }

    /**
     * Build the input array for Part 4 by mapping interpreter input tensor names
     * (serving_default_args_N:0) to the correct ByteBuffer.
     *
     * This is necessary because converter/export pipelines can reorder or drop inputs,
     * so relying on a hard-coded positional order can break.
     */
    private fun buildPart4Inputs(
        interpreter: Interpreter,
        image: ByteBuffer,
        imageTokens: ByteBuffer,
        latent0: ByteBuffer,
        latent1: ByteBuffer,
        x0Feat: ByteBuffer,
        x1Feat: ByteBuffer,
        x2Feat: ByteBuffer
    ): Array<Any> {
        val byArgIndex: Map<Int, ByteBuffer> = mapOf(
            0 to image,
            1 to imageTokens,
            2 to latent0,
            3 to latent1,
            4 to x0Feat,
            5 to x1Feat,
            6 to x2Feat
        )

        val inputCount = interpreter.inputTensorCount
        val inputs = arrayOfNulls<Any>(inputCount)

        Log.d(TAG, "Part 4 interpreter inputs ($inputCount):")
        for (i in 0 until inputCount) {
            val t = interpreter.getInputTensor(i)
            val name = t.name()
            val shape = t.shape().contentToString()
            val bytes = t.numBytes()
            Log.d(TAG, "  input[$i] name=$name shape=$shape bytes=$bytes")

            val argIdx = parseServingArgIndex(name)
            val buf = argIdx?.let { byArgIndex[it] }
            if (buf != null) {
                inputs[i] = buf
            }
        }

        // Fallback: fill any missing inputs by shape match (rare, but safer)
        val remaining = mutableListOf(
            image to intArrayOf(1, 3, IMAGE_SIZE, IMAGE_SIZE),
            imageTokens to intArrayOf(1, 577, FEATURE_DIM),
            latent0 to intArrayOf(1, FEATURE_DIM, 96, 96),
            latent1 to intArrayOf(1, FEATURE_DIM, 96, 96),
            x0Feat to intArrayOf(1, FEATURE_DIM, 96, 96),
            x1Feat to intArrayOf(1, FEATURE_DIM, 48, 48),
            x2Feat to intArrayOf(1, FEATURE_DIM, 24, 24),
        )
        fun shapeEquals(a: IntArray, b: IntArray): Boolean = a.size == b.size && a.indices.all { a[it] == b[it] }

        for (i in 0 until inputCount) {
            if (inputs[i] != null) continue
            val t = interpreter.getInputTensor(i)
            val tShape = t.shape()
            val candidate = remaining.firstOrNull { (_, s) -> shapeEquals(tShape, s) }
            if (candidate != null) {
                inputs[i] = candidate.first
                remaining.remove(candidate)
                Log.w(TAG, "Part 4: filled input[$i] by shape match (name=${t.name()})")
            }
        }

        val missing = inputs.indexOfFirst { it == null }
        if (missing != -1) {
            Log.e(TAG, "Part 4: could not map all inputs; missing index $missing")
        }

        @Suppress("UNCHECKED_CAST")
        return inputs as Array<Any>
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
            val interpreter = createInterpreter(modelFile, numThreadsOverride = 4, useXnnpack = true)

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
            // Step 5: Write PLY
            progressCallback?.invoke(0.70f, "Writing PLY ($gaussianCount Gaussians)...")
            val result = writeGaussianPlyFromPackedBuffer(outputBuffer, gaussianCount, progressCallback)

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
     *
     * Optimized: single pass over pixels + bulk FloatBuffer.put().
     * Old approach: 3 separate passes with per-pixel putFloat() calls (~7M calls).
     * New approach: 1 pass into FloatArray + 1 bulk put (~2.36M iterations + 1 memcpy).
     * This mirrors iOS's Accelerate vImage-style batch processing.
     */
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val pixelCount = width * height
        val pixels = IntArray(pixelCount)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // Single-pass: extract R, G, B channels simultaneously into CHW layout
        val channelFloats = FloatArray(3 * pixelCount)
        val inv255 = 1f / 255f
        for (i in 0 until pixelCount) {
            val pixel = pixels[i]
            channelFloats[i] = ((pixel shr 16) and 0xFF) * inv255                  // R channel
            channelFloats[pixelCount + i] = ((pixel shr 8) and 0xFF) * inv255      // G channel
            channelFloats[2 * pixelCount + i] = (pixel and 0xFF) * inv255           // B channel
        }

        // Bulk write — one native memcpy instead of ~7M individual putFloat calls
        val buffer = ByteBuffer.allocateDirect(channelFloats.size * 4).apply {
            order(ByteOrder.nativeOrder())
        }
        buffer.asFloatBuffer().put(channelFloats)
        buffer.rewind()
        return buffer
    }

    /**
     * Write Gaussians to PLY file from interleaved [N, 14] format (FloatArray path).
     * Same BLAS-like optimizations as writeGaussianPlyFromPackedBuffer:
     * - Sigmoid LUT, zero SH block, LN/logit LUTs
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
            val logitLutScale = (LOGIT_LUT_SIZE - 1).toFloat()

            val progressEvery = max(1, gaussianCount / 10)
            var processed = 0

            while (processed < gaussianCount) {
                val currentBatch = minOf(batchSize, gaussianCount - processed)

                for (j in 0 until currentBatch) {
                    val offset = (processed + j) * PARAMS_PER_GAUSSIAN

                    val posX = gaussianParams[offset]
                    val posY = -gaussianParams[offset + 1]
                    val posZ = -gaussianParams[offset + 2]

                    if (posX < minX) minX = posX; if (posX > maxX) maxX = posX
                    if (posY < minY) minY = posY; if (posY > maxY) maxY = posY
                    if (posZ < minZ) minZ = posZ; if (posZ > maxZ) maxZ = posZ

                    batchBuffer.putFloat(posX)
                    batchBuffer.putFloat(posY)
                    batchBuffer.putFloat(posZ)

                    batchBuffer.putFloat(0f)
                    batchBuffer.putFloat(0f)
                    batchBuffer.putFloat(0f)

                    // BGR->RGB swap (TFLite outputs BGR; ONNX outputs RGB - match ONNX)
                    val colorR = gaussianParams[offset + 13].coerceIn(0f, 1f)
                    val colorG = gaussianParams[offset + 12].coerceIn(0f, 1f)
                    val colorB = gaussianParams[offset + 11].coerceIn(0f, 1f)
                    batchBuffer.putFloat((colorR - 0.5f) / SH_C0)
                    batchBuffer.putFloat((colorG - 0.5f) / SH_C0)
                    batchBuffer.putFloat((colorB - 0.5f) / SH_C0)

                    // Bulk zero SH block instead of 45 individual putFloat(0f)
                    batchBuffer.put(ZERO_SH_BLOCK)

                    // Sigmoid via LUT (no Math.exp)
                    val rawOpacityLogit = gaussianParams[offset + 3]
                    val opacitySigmoid = sigmoidLut(rawOpacityLogit)
                    val opacityLutIndex = (opacitySigmoid * logitLutScale).toInt()
                        .coerceIn(0, LOGIT_LUT_SIZE - 1)
                    batchBuffer.putFloat(LOGIT_LUT[opacityLutIndex])

                    batchBuffer.putFloat(lnLut(max(gaussianParams[offset + 4] * scaleBoost, minScale)))
                    batchBuffer.putFloat(lnLut(max(gaussianParams[offset + 5] * scaleBoost, minScale)))
                    batchBuffer.putFloat(lnLut(max(gaussianParams[offset + 6] * scaleBoost, minScale)))

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

    /**
     * Memory-safe PLY writer: consumes packed float output directly from a direct ByteBuffer
     * without copying into a large FloatArray on the Java heap.
     *
     * Packed format per-gaussian: 14 floats
     * [x, y, z, opacity_logit, s0, s1, s2, qw, qx, qy, qz, r, g, b]
     *
     * BLAS-like optimizations (matching ONNX version and iOS Accelerate patterns):
     * 1. Bulk FloatBuffer.get() reads batchSize*14 floats in one JNI call (like cblas_scopy)
     *    instead of 14 individual get() calls per Gaussian
     * 2. Pre-computed sigmoid LUT replaces Math.exp() per Gaussian (like vvexpf + vvdivf)
     * 3. Pre-allocated zero SH block replaces 45 individual putFloat(0f) calls (like memset)
     * 4. Local FloatArray processing avoids per-element JNI/buffer overhead
     */
    private fun writeGaussianPlyFromPackedBuffer(
        packedBuffer: ByteBuffer,
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

        val floatBuf = packedBuffer
            .duplicate()
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        floatBuf.rewind()

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
            val logitLutScale = (LOGIT_LUT_SIZE - 1).toFloat()

            // Pre-allocated local array for vectorized bulk reads — one JNI call per batch
            // instead of batchSize × 14 individual JNI calls. This is the key BLAS-like
            // optimization from the ONNX PLY writer.
            val localPacked = FloatArray(batchSize * PARAMS_PER_GAUSSIAN)

            val progressEvery = max(1, gaussianCount / 10)
            var processed = 0

            while (processed < gaussianCount) {
                val currentBatch = minOf(batchSize, gaussianCount - processed)
                val floatsToRead = currentBatch * PARAMS_PER_GAUSSIAN

                // BLAS-like bulk read: one JNI call reads up to 7168 floats (512×14)
                floatBuf.get(localPacked, 0, floatsToRead)

                // Process from local array — no per-element JNI/buffer overhead
                for (j in 0 until currentBatch) {
                    val baseIdx = j * PARAMS_PER_GAUSSIAN

                    val posX = localPacked[baseIdx]
                    val posY = -localPacked[baseIdx + 1]
                    val posZ = -localPacked[baseIdx + 2]

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

                    // Color (offset 11-13) → SH DC; BGR->RGB swap (match ONNX)
                    val colorR = localPacked[baseIdx + 13].coerceIn(0f, 1f)
                    val colorG = localPacked[baseIdx + 12].coerceIn(0f, 1f)
                    val colorB = localPacked[baseIdx + 11].coerceIn(0f, 1f)
                    batchBuffer.putFloat((colorR - 0.5f) / SH_C0)
                    batchBuffer.putFloat((colorG - 0.5f) / SH_C0)
                    batchBuffer.putFloat((colorB - 0.5f) / SH_C0)

                    // Higher order SH (45 zeros) — bulk put instead of 45 individual putFloat(0f)
                    batchBuffer.put(ZERO_SH_BLOCK)

                    // Opacity: sigmoid via LUT (avoids Math.exp per Gaussian)
                    val rawOpacityLogit = localPacked[baseIdx + 3]
                    val opacitySigmoid = sigmoidLut(rawOpacityLogit)
                    val opacityLutIndex = (opacitySigmoid * logitLutScale).toInt()
                        .coerceIn(0, LOGIT_LUT_SIZE - 1)
                    batchBuffer.putFloat(LOGIT_LUT[opacityLutIndex])

                    // Scale (offset 4-6) → log transform via LUT
                    batchBuffer.putFloat(lnLut(max(localPacked[baseIdx + 4] * scaleBoost, minScale)))
                    batchBuffer.putFloat(lnLut(max(localPacked[baseIdx + 5] * scaleBoost, minScale)))
                    batchBuffer.putFloat(lnLut(max(localPacked[baseIdx + 6] * scaleBoost, minScale)))

                    // Rotation quaternion (offset 7-10) → normalize
                    val quatW = localPacked[baseIdx + 7]
                    val quatX = localPacked[baseIdx + 8]
                    val quatY = localPacked[baseIdx + 9]
                    val quatZ = localPacked[baseIdx + 10]
                    val quatMag = sqrt(quatW * quatW + quatX * quatX + quatY * quatY + quatZ * quatZ)
                    val quatInvMag = if (quatMag > 1e-8f) 1f / quatMag else 1f
                    batchBuffer.putFloat(quatW * quatInvMag)
                    batchBuffer.putFloat(quatX * quatInvMag)
                    batchBuffer.putFloat(quatY * quatInvMag)
                    batchBuffer.putFloat(quatZ * quatInvMag)
                }

                // Flush batch to disk via zero-copy channel write
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
