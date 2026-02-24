package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
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
 * ExecuTorch INT8 SHARP inference using a single quantized .pte model.
 *
 * The INT8 model runs the full pipeline in one forward pass:
 *   [1, 3, 1536, 1536] image -> [N, 14] Gaussian parameters
 *
 * No split parts, no chunking. PT2E INT8 quantization with XNNPACK targets
 * ARM NEON INT8 kernels, reducing data movement by ~4x vs FP32.
 */
class ExecutorchInt8Sharp private constructor(private val context: Context) {

    companion object {
        private const val TAG = "ExecutorchInt8Sharp"
        private const val INPUT_SIZE = 1536
        private const val PATCH_SIZE = 384
        private const val FEATURE_DIM = 1024
        private const val SPATIAL_SIZE = 24
        private const val PARAMS_PER_GAUSSIAN = 14
        private const val SH_C0 = 0.28209479177387814f
        private const val FLOATS_PER_VERTEX = 62
        private const val BYTES_PER_VERTEX = FLOATS_PER_VERTEX * 4
        private const val PLY_BATCH_SIZE = 512

        private const val GRID_1X = 5
        private const val GRID_05X = 3
        private const val TOTAL_PATCHES = 35
        private const val PADDING_1X = 3
        private const val PADDING_05X = 6

        private const val SPLIT_PART1 = "sharp_split_part1_int8.pte"
        private const val SPLIT_PART2 = "sharp_split_part2_int8.pte"
        private const val SPLIT_PART3 = "sharp_split_part3_int8.pte"
        private const val SPLIT_PART4 = "sharp_split_part4_int8.pte"
        private val SPLIT_FILENAMES = arrayOf(SPLIT_PART1, SPLIT_PART2, SPLIT_PART3, SPLIT_PART4)

        // Chunked Part 4: ViT chunks (FP32) + decoder (FP32). Used instead of monolithic Part 4
        // because the decoder's ~4GB activation memory exceeds device RAM in a single forward.
        private const val PART4A_CHUNK_512 = "sharp_split_part4a_chunk_512.pte"
        private const val PART4A_CHUNK_65 = "sharp_split_part4a_chunk_65.pte"
        private const val PART4B = "sharp_split_part4b.pte"

        private const val MODEL_FILENAME = "sharp_int8.pte"

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
        private var instance: ExecutorchInt8Sharp? = null

        fun getInstance(context: Context): ExecutorchInt8Sharp {
            return instance ?: synchronized(this) {
                instance ?: ExecutorchInt8Sharp(context.applicationContext).also {
                    instance = it
                    Log.d(TAG, "ExecutorchInt8Sharp singleton created")
                }
            }
        }
    }

    private val mutex = Mutex()

    @Volatile
    private var isInitialized = false

    private val modelsDir: File by lazy {
        context.getExternalFilesDir("models") ?: File(context.filesDir, "models")
    }

    private val internalModelsDir: File by lazy {
        File(context.filesDir, "models").also { it.mkdirs() }
    }

    private val plyBatch: ByteBuffer by lazy {
        ByteBuffer.allocateDirect(BYTES_PER_VERTEX * PLY_BATCH_SIZE).apply { order(ByteOrder.LITTLE_ENDIAN) }
    }
    private val zeroSHBuffer: ByteBuffer by lazy {
        ByteBuffer.allocateDirect(45 * 4).apply { order(ByteOrder.LITTLE_ENDIAN) }
    }

    data class StreamingResult(
        val plyFile: File,
        val classicPlyFile: File,
        val gaussianCount: Int,
        val roomWidth: Float,
        val roomHeight: Float,
        val roomDepth: Float
    )

    private fun isChunkedPart4Available(): Boolean =
        findFile(PART4A_CHUNK_512) != null && findFile(PART4A_CHUNK_65) != null && findFile(PART4B) != null

    fun isModelReady(): Boolean {
        // INT8 Parts 1-3 + chunked Part 4 (FP32)
        val int8EncodersReady = findFile(SPLIT_PART1) != null && findFile(SPLIT_PART2) != null && findFile(SPLIT_PART3) != null
        if (int8EncodersReady && isChunkedPart4Available()) {
            Log.d(TAG, "isModelReady: INT8 encoders (3 parts) + chunked Part 4 (3 chunks)")
            return true
        }
        // INT8 Parts 1-4 (monolithic Part 4, may OOM)
        val splitReady = SPLIT_FILENAMES.all { findFile(it) != null }
        if (splitReady) {
            Log.d(TAG, "isModelReady: INT8 split models found (${SPLIT_FILENAMES.size} parts, monolithic Part 4)")
            return true
        }
        val singleReady = findFile(MODEL_FILENAME) != null
        if (singleReady) {
            Log.d(TAG, "isModelReady: INT8 single model found")
        } else {
            Log.d(TAG, "isModelReady: no INT8 models found")
        }
        return singleReady
    }

    private fun isSplitMode(): Boolean {
        val int8Encoders = findFile(SPLIT_PART1) != null && findFile(SPLIT_PART2) != null && findFile(SPLIT_PART3) != null
        return int8Encoders && (isChunkedPart4Available() || findFile(SPLIT_PART4) != null)
    }

    fun getMissingFiles(): List<String> {
        if (isSplitMode()) return emptyList()
        val missing = SPLIT_FILENAMES.filter { findFile(it) == null }.toMutableList()
        if (findFile(MODEL_FILENAME) == null) missing.add(MODEL_FILENAME)
        return missing
    }

    fun getModelsDirPath(): String = modelsDir.absolutePath

    fun initialize(): Boolean = runBlocking { mutex.withLock { initializeImpl() } }

    private fun initializeImpl(): Boolean {
        Log.d(TAG, "initialize ENTER. Memory: ${getMemoryInfo()}")
        if (isSplitMode()) {
            for (fn in SPLIT_FILENAMES) {
                val f = findFile(fn)!!
                Log.d(TAG, "  INT8 split: ${f.name} (${f.length() / 1024 / 1024}MB)")
            }
            isInitialized = true
            return true
        }
        val modelFile = findFile(MODEL_FILENAME)
        if (modelFile != null) {
            Log.d(TAG, "  INT8 single: ${modelFile.name} (${modelFile.length() / 1024 / 1024}MB)")
            isInitialized = true
            return true
        }
        Log.e(TAG, "No INT8 models found")
        return false
    }

    suspend fun preloadAndWarmup(progress: ((String) -> Unit)? = null) = withContext(Dispatchers.IO) {
        mutex.withLock {
            if (!isInitialized) initializeImpl()
        }
        val preloadFile = if (isSplitMode()) findFile(SPLIT_PART1) else findFile(MODEL_FILENAME)
        if (preloadFile == null) return@withContext
        val label = if (isSplitMode()) "INT8 Part1" else "INT8 single"
        Log.d(TAG, "preloadAndWarmup: validating $label (${preloadFile.length() / 1024 / 1024}MB)")
        progress?.invoke("Validating $label...")
        val t0 = System.currentTimeMillis()
        val module = Module.load(preloadFile.absolutePath, Module.LOAD_MODE_MMAP)
        Log.d(TAG, "PRELOAD $label load ${System.currentTimeMillis() - t0}ms")
        module.destroy()
        System.gc()
        Log.d(TAG, "preloadAndWarmup DONE")
        progress?.invoke("Preload done")
    }

    suspend fun inferStreaming(
        bitmap: Bitmap,
        progressCallback: ((Float, String) -> Unit)? = null
    ): StreamingResult? = withContext(Dispatchers.IO) {
        mutex.withLock {
            if (!isInitialized) {
                Log.e(TAG, "Not initialized")
                return@withContext null
            }
            Log.d(TAG, "inferStreaming ENTER (INT8) bitmap=${bitmap.width}x${bitmap.height} split=${isSplitMode()} Memory: ${getMemoryInfo()}")
            return@withContext if (isSplitMode()) {
                inferSplitMode(bitmap, progressCallback)
            } else {
                inferSingleModel(bitmap, progressCallback)
            }
        }
    }

    /**
     * Split 4-part INT8 inference. Each part loaded/run/destroyed sequentially.
     * Intermediate tensors stay in RAM (no disk I/O between parts) to avoid
     * synchronization drift that causes "snake-like" spatial artifacts.
     * INT8 models accept FP32 input — quantization/dequantization is internal to the .pte.
     */
    private fun inferSplitMode(
        bitmap: Bitmap,
        progressCallback: ((Float, String) -> Unit)?
    ): StreamingResult? {

        // Exact port of Python merge() from spn_encoder.py:
        // Crop overlapping borders, then concatenate. No blending/averaging.
        fun mergePatchCropInto(
            output: FloatArray, outW: Int, patch: FloatArray,
            gridI: Int, gridJ: Int, gridSize: Int, padding: Int,
            outOffsetY: IntArray, outOffsetX: IntArray
        ) {
            val pH = SPATIAL_SIZE; val pW = SPATIAL_SIZE; val C = FEATURE_DIM
            val pHW = pH * pW

            // Crop borders exactly like Python: trim padding from non-edge sides
            val cropTop = if (gridJ != 0) padding else 0
            val cropBot = if (gridJ != gridSize - 1) padding else 0
            val cropLeft = if (gridI != 0) padding else 0
            val cropRight = if (gridI != gridSize - 1) padding else 0
            val croppedH = pH - cropTop - cropBot
            val croppedW = pW - cropLeft - cropRight

            val dstY = outOffsetY[0]
            val dstX = outOffsetX[gridI]
            val outHW = outW * outW

            for (c in 0 until C) {
                val srcBase = c * pHW
                val dstBase = c * outHW
                for (dy in 0 until croppedH) {
                    for (dx in 0 until croppedW) {
                        val srcIdx = srcBase + (cropTop + dy) * pW + (cropLeft + dx)
                        val dstIdx = dstBase + (dstY + dy) * outW + (dstX + dx)
                        output[dstIdx] = patch[srcIdx]
                    }
                }
            }

            // Advance row offset after last column of this row
            if (gridI == gridSize - 1) {
                outOffsetY[0] += croppedH
            }
        }

        fun buildColumnOffsets(gridSize: Int, padding: Int): IntArray {
            val offsets = IntArray(gridSize)
            var x = 0
            for (col in 0 until gridSize) {
                offsets[col] = x
                val cropLeft = if (col != 0) padding else 0
                val cropRight = if (col != gridSize - 1) padding else 0
                x += SPATIAL_SIZE - cropLeft - cropRight
            }
            return offsets
        }

        fun reshapeTokensToSpatial(tokens: FloatArray, spatialOut: FloatArray) {
            val hw = SPATIAL_SIZE * SPATIAL_SIZE
            for (h in 0 until SPATIAL_SIZE) {
                for (w in 0 until SPATIAL_SIZE) {
                    val seqIdx = 1 + h * SPATIAL_SIZE + w
                    val dstPix = h * SPATIAL_SIZE + w
                    val srcBase = seqIdx * FEATURE_DIM
                    for (c in 0 until FEATURE_DIM) {
                        spatialOut[c * hw + dstPix] = tokens[srcBase + c]
                    }
                }
            }
        }

        fun getMergedSize(gridSize: Int, patchSpatial: Int, padding: Int): Int {
            val border = patchSpatial - padding
            val inner = patchSpatial - 2 * padding
            return 2 * border + (gridSize - 2) * inner
        }

        fun preprocessPatch(patchBitmap: Bitmap): FloatArray {
            val w = patchBitmap.width; val h = patchBitmap.height; val n = w * h
            val pixels = IntArray(n)
            patchBitmap.getPixels(pixels, 0, w, 0, 0, w, h)
            val data = FloatArray(3 * n)
            val inv255 = 1f / 255f
            for (i in 0 until n) {
                val p = pixels[i]
                data[i] = ((p shr 16) and 0xFF) * inv255
                data[n + i] = ((p shr 8) and 0xFF) * inv255
                data[2 * n + i] = (p and 0xFF) * inv255
            }
            return data
        }

        val startTime = System.currentTimeMillis()
        try {
            Log.d(TAG, "inferSplitMode ENTER (INT8 split) Memory: ${getMemoryInfo()}")

            // Resize to native 1536x1536 BEFORE any inference to ensure consistent tensor sizes
            progressCallback?.invoke(0.02f, "Preprocessing to ${INPUT_SIZE}x${INPUT_SIZE}...")
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
            Log.d(TAG, "Bitmap resized: ${bitmap.width}x${bitmap.height} -> ${INPUT_SIZE}x${INPUT_SIZE}")
            val halfSize = INPUT_SIZE / 2
            val halfBitmap = Bitmap.createScaledBitmap(scaledBitmap, halfSize, halfSize, true)
            val quarterBitmap = Bitmap.createScaledBitmap(scaledBitmap, PATCH_SIZE, PATCH_SIZE, true)

            val mergedSize1x = getMergedSize(GRID_1X, SPATIAL_SIZE, PADDING_1X)
            val mergedSize05x = getMergedSize(GRID_05X, SPATIAL_SIZE, PADDING_05X)
            Log.d(TAG, "Merged sizes: 1x=${mergedSize1x}x${mergedSize1x}, 0.5x=${mergedSize05x}x${mergedSize05x}")
            val stride1x = (INPUT_SIZE - PATCH_SIZE) / 4
            val stride05x = (halfSize - PATCH_SIZE) / 2
            val tempSpatial = FloatArray(FEATURE_DIM * SPATIAL_SIZE * SPATIAL_SIZE)

            // Merged feature maps (all in RAM — no disk I/O between parts)
            val latent0 = FloatArray(FEATURE_DIM * mergedSize1x * mergedSize1x)
            val latent1 = FloatArray(FEATURE_DIM * mergedSize1x * mergedSize1x)
            val x0Feat = FloatArray(FEATURE_DIM * mergedSize1x * mergedSize1x)
            val x1Feat = FloatArray(FEATURE_DIM * mergedSize05x * mergedSize05x)
            var x2Feat: FloatArray? = null
            // Row/column offsets for crop-and-concatenate merge (no blending)
            val colOffsets1x = buildColumnOffsets(GRID_1X, PADDING_1X)
            val colOffsets05x = buildColumnOffsets(GRID_05X, PADDING_05X)
            val rowOffset1xL0 = intArrayOf(0)
            val rowOffset1xL1 = intArrayOf(0)
            val rowOffset1xX0 = intArrayOf(0)
            val rowOffset05x = intArrayOf(0)

            // --- Part 1 + Part 2: 35 patches, merge on-the-fly in RAM ---
            progressCallback?.invoke(0.05f, "Loading INT8 encoders...")
            val part1File = findFile(SPLIT_PART1)!!
            val part2File = findFile(SPLIT_PART2)!!
            val module1 = Module.load(part1File.absolutePath, Module.LOAD_MODE_MMAP)
            val module2 = Module.load(part2File.absolutePath, Module.LOAD_MODE_MMAP)
            Log.d(TAG, "INT8 Part1+Part2 loaded (mmap). Memory: ${getMemoryInfo()}")

            // Warm-up: stabilize XNNPACK threadpool and CPU clock before the 35-patch loop
            val warmupPatch = FloatArray(3 * PATCH_SIZE * PATCH_SIZE)
            val warmupTensor = Tensor.fromBlob(warmupPatch, longArrayOf(1, 3, PATCH_SIZE.toLong(), PATCH_SIZE.toLong()))
            module1.forward(EValue.from(warmupTensor))
            Log.d(TAG, "Warm-up forward done. Memory: ${getMemoryInfo()}")

            var patchCount = 0
            val p12Start = System.currentTimeMillis()

            // 1x scale patches (5x5 = 25)
            for (i in 0 until GRID_1X) {
                for (j in 0 until GRID_1X) {
                    val patchBitmap = Bitmap.createBitmap(scaledBitmap, j * stride1x, i * stride1x, PATCH_SIZE, PATCH_SIZE)
                    val patchData = preprocessPatch(patchBitmap)
                    patchBitmap.recycle()

                    val inputTensor = Tensor.fromBlob(patchData, longArrayOf(1, 3, PATCH_SIZE.toLong(), PATCH_SIZE.toLong()))
                    val out1 = module1.forward(EValue.from(inputTensor))
                    val tokens = out1[0].toTensor().dataAsFloatArray
                    val block5 = out1[1].toTensor().dataAsFloatArray

                    reshapeTokensToSpatial(block5, tempSpatial)
                    mergePatchCropInto(latent0, mergedSize1x, tempSpatial, j, i, GRID_1X, PADDING_1X, rowOffset1xL0, colOffsets1x)
                    reshapeTokensToSpatial(tokens, tempSpatial)
                    mergePatchCropInto(latent1, mergedSize1x, tempSpatial, j, i, GRID_1X, PADDING_1X, rowOffset1xL1, colOffsets1x)

                    val tokensTensor = Tensor.fromBlob(tokens, longArrayOf(1, 577, 1024))
                    val feat = module2.forward(EValue.from(tokensTensor))[0].toTensor().dataAsFloatArray
                    mergePatchCropInto(x0Feat, mergedSize1x, feat, j, i, GRID_1X, PADDING_1X, rowOffset1xX0, colOffsets1x)

                    patchCount++
                    progressCallback?.invoke(0.05f + (patchCount.toFloat() / TOTAL_PATCHES) * 0.40f,
                        "INT8 Part 1+2: Patch $patchCount/$TOTAL_PATCHES...")
                }
            }
            // 0.5x scale patches (3x3 = 9)
            for (i in 0 until GRID_05X) {
                for (j in 0 until GRID_05X) {
                    val patchBitmap = Bitmap.createBitmap(halfBitmap, j * stride05x, i * stride05x, PATCH_SIZE, PATCH_SIZE)
                    val patchData = preprocessPatch(patchBitmap)
                    patchBitmap.recycle()

                    val inputTensor = Tensor.fromBlob(patchData, longArrayOf(1, 3, PATCH_SIZE.toLong(), PATCH_SIZE.toLong()))
                    val tokens = module1.forward(EValue.from(inputTensor))[0].toTensor().dataAsFloatArray
                    val tokensTensor = Tensor.fromBlob(tokens, longArrayOf(1, 577, 1024))
                    val feat = module2.forward(EValue.from(tokensTensor))[0].toTensor().dataAsFloatArray
                    mergePatchCropInto(x1Feat, mergedSize05x, feat, j, i, GRID_05X, PADDING_05X, rowOffset05x, colOffsets05x)
                    patchCount++
                    progressCallback?.invoke(0.05f + (patchCount.toFloat() / TOTAL_PATCHES) * 0.40f,
                        "INT8 Part 1+2: Patch $patchCount/$TOTAL_PATCHES...")
                }
            }
            halfBitmap.recycle()

            // 0.25x scale patch (1x1 = 1)
            val qData = preprocessPatch(quarterBitmap)
            quarterBitmap.recycle()
            val qInput = Tensor.fromBlob(qData, longArrayOf(1, 3, PATCH_SIZE.toLong(), PATCH_SIZE.toLong()))
            val qTokens = module1.forward(EValue.from(qInput))[0].toTensor().dataAsFloatArray
            val qTokensTensor = Tensor.fromBlob(qTokens, longArrayOf(1, 577, 1024))
            x2Feat = module2.forward(EValue.from(qTokensTensor))[0].toTensor().dataAsFloatArray
            patchCount++
            progressCallback?.invoke(0.05f + (patchCount.toFloat() / TOTAL_PATCHES) * 0.40f,
                "INT8 Part 1+2: Patch $patchCount/$TOTAL_PATCHES...")

            module1.destroy(); module2.destroy()
            System.gc()
            System.runFinalization()
            val p12Time = System.currentTimeMillis() - p12Start
            Log.d(TAG, "INT8 Part 1+2 done: $patchCount patches in ${p12Time}ms Memory: ${getMemoryInfo()}")

            // Precompute full image data before Part 3 load, then recycle bitmap to free ~28MB
            val imageData = preprocessImage(scaledBitmap)
            scaledBitmap.recycle()

            // --- Part 3: Image encoder (full image) ---
            progressCallback?.invoke(0.55f, "INT8 Part 3: Image encoder...")
            val part3File = findFile(SPLIT_PART3)!!
            val module3 = Module.load(part3File.absolutePath, Module.LOAD_MODE_MMAP)
            val fullImageTensor = Tensor.fromBlob(imageData, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))

            val p3Start = System.currentTimeMillis()
            val imageTokensData = module3.forward(EValue.from(fullImageTensor))[0].toTensor().dataAsFloatArray
            val p3Time = System.currentTimeMillis() - p3Start
            module3.destroy()
            System.gc()
            System.runFinalization()
            Thread.sleep(200)
            Log.d(TAG, "INT8 Part 3 done in ${p3Time}ms Memory: ${getMemoryInfo()}")

            // --- Part 4: Chunked (4a_512 + 4a_65 + 4b) to avoid decoder OOM ---
            // Each chunk is loaded, run, destroyed sequentially. Peak memory = max(single_chunk).
            System.gc(); System.runFinalization(); Thread.sleep(200)
            Log.d(TAG, "Pre-Part4 memory after GC: ${getMemoryInfo()}")

            val useChunked = isChunkedPart4Available()
            Log.d(TAG, "Part 4: chunked=$useChunked. Memory: ${getMemoryInfo()}")
            val p4Start = System.currentTimeMillis()

            val outputData: FloatArray
            val gaussianCount: Int

            if (useChunked) {
                // Slice imageTokens [1,577,1024] into [1,512,1024] and [1,65,1024]
                val tokens512 = FloatArray(512 * FEATURE_DIM)
                val tokens65 = FloatArray(65 * FEATURE_DIM)
                System.arraycopy(imageTokensData, 0, tokens512, 0, 512 * FEATURE_DIM)
                System.arraycopy(imageTokensData, 512 * FEATURE_DIM, tokens65, 0, 65 * FEATURE_DIM)

                // --- Part 4a chunk 512: ViT blocks 12-23 on first 512 tokens ---
                progressCallback?.invoke(0.65f, "INT8 Part 4a (512 tokens)...")
                val mod4a512 = Module.load(findFile(PART4A_CHUNK_512)!!.absolutePath, Module.LOAD_MODE_MMAP)
                val tensor512 = Tensor.fromBlob(tokens512, longArrayOf(1, 512, FEATURE_DIM.toLong()))
                val chunk512Out = mod4a512.forward(EValue.from(tensor512))[0].toTensor().dataAsFloatArray
                mod4a512.destroy(); System.gc()
                Log.d(TAG, "Part 4a (512) done. Output: ${chunk512Out.size} floats. Memory: ${getMemoryInfo()}")

                // --- Part 4a chunk 65: ViT blocks 12-23 on remaining 65 tokens ---
                progressCallback?.invoke(0.70f, "INT8 Part 4a (65 tokens)...")
                val mod4a65 = Module.load(findFile(PART4A_CHUNK_65)!!.absolutePath, Module.LOAD_MODE_MMAP)
                val tensor65 = Tensor.fromBlob(tokens65, longArrayOf(1, 65, FEATURE_DIM.toLong()))
                val chunk65Out = mod4a65.forward(EValue.from(tensor65))[0].toTensor().dataAsFloatArray
                mod4a65.destroy(); System.gc()
                Log.d(TAG, "Part 4a (65) done. Output: ${chunk65Out.size} floats. Memory: ${getMemoryInfo()}")

                // Concatenate normalized token chunks -> [1, 577, 1024]
                val tokensNorm = FloatArray(577 * FEATURE_DIM)
                System.arraycopy(chunk512Out, 0, tokensNorm, 0, chunk512Out.size)
                System.arraycopy(chunk65Out, 0, tokensNorm, chunk512Out.size, chunk65Out.size)

                // --- Part 4b: Decoder (FP32, ~178MB, fits in memory) ---
                progressCallback?.invoke(0.75f, "INT8 Part 4b: Decoder...")
                System.gc(); System.runFinalization(); Thread.sleep(150)
                val mod4b = Module.load(findFile(PART4B)!!.absolutePath, Module.LOAD_MODE_MMAP)
                Log.d(TAG, "Part 4b loaded (${findFile(PART4B)!!.length() / 1024 / 1024}MB). Memory: ${getMemoryInfo()}")

                val p4bOut = mod4b.forward(
                    EValue.from(Tensor.fromBlob(tokensNorm, longArrayOf(1, 577, FEATURE_DIM.toLong()))),
                    EValue.from(Tensor.fromBlob(imageData, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))),
                    EValue.from(Tensor.fromBlob(latent0, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize1x.toLong(), mergedSize1x.toLong()))),
                    EValue.from(Tensor.fromBlob(latent1, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize1x.toLong(), mergedSize1x.toLong()))),
                    EValue.from(Tensor.fromBlob(x0Feat, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize1x.toLong(), mergedSize1x.toLong()))),
                    EValue.from(Tensor.fromBlob(x1Feat, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize05x.toLong(), mergedSize05x.toLong()))),
                    EValue.from(Tensor.fromBlob(x2Feat!!, longArrayOf(1, FEATURE_DIM.toLong(), SPATIAL_SIZE.toLong(), SPATIAL_SIZE.toLong())))
                )

                val outTensor = p4bOut[0].toTensor()
                gaussianCount = (outTensor.numel() / PARAMS_PER_GAUSSIAN).toInt()
                outputData = outTensor.dataAsFloatArray
                mod4b.destroy(); System.gc()

            } else {
                // Monolithic Part 4 fallback (may OOM on decoder activations)
                progressCallback?.invoke(0.65f, "INT8 Part 4: Decoder (monolithic)...")
                val part4File = findFile(SPLIT_PART4)!!
                val module4 = Module.load(part4File.absolutePath, Module.LOAD_MODE_MMAP)
                Log.d(TAG, "Part 4 loaded (${part4File.length() / 1024 / 1024}MB). Memory: ${getMemoryInfo()}")

                val p4Out = module4.forward(
                    EValue.from(Tensor.fromBlob(imageTokensData, longArrayOf(1, 577, FEATURE_DIM.toLong()))),
                    EValue.from(Tensor.fromBlob(imageData, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))),
                    EValue.from(Tensor.fromBlob(latent0, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize1x.toLong(), mergedSize1x.toLong()))),
                    EValue.from(Tensor.fromBlob(latent1, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize1x.toLong(), mergedSize1x.toLong()))),
                    EValue.from(Tensor.fromBlob(x0Feat, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize1x.toLong(), mergedSize1x.toLong()))),
                    EValue.from(Tensor.fromBlob(x1Feat, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize05x.toLong(), mergedSize05x.toLong()))),
                    EValue.from(Tensor.fromBlob(x2Feat!!, longArrayOf(1, FEATURE_DIM.toLong(), SPATIAL_SIZE.toLong(), SPATIAL_SIZE.toLong())))
                )

                val outTensor = p4Out[0].toTensor()
                gaussianCount = (outTensor.numel() / PARAMS_PER_GAUSSIAN).toInt()
                outputData = outTensor.dataAsFloatArray
                module4.destroy(); System.gc()
            }

            val p4Time = System.currentTimeMillis() - p4Start
            Log.d(TAG, "INT8 Part 4 done in ${p4Time}ms (chunked=$useChunked). Memory: ${getMemoryInfo()}")

            progressCallback?.invoke(0.80f, "Writing PLY ($gaussianCount Gaussians)...")
            val result = writeStreamingPly(gaussianCount, FloatBuffer.wrap(outputData), progressCallback)

            val totalTime = System.currentTimeMillis() - startTime
            Log.d(TAG, "INT8 split-mode COMPLETE: ${totalTime}ms gaussianCount=${result.gaussianCount}")
            Log.d(TAG, "  Breakdown: P1+P2=${p12Time}ms P3=${p3Time}ms P4=${p4Time}ms")
            Log.d(TAG, "  Room: ${result.roomWidth}m x ${result.roomHeight}m x ${result.roomDepth}m")

            progressCallback?.invoke(1.0f, "Done!")
            return result

        } catch (e: Exception) {
            Log.e(TAG, "INT8 split inference failed: ${e.message}", e)
            progressCallback?.invoke(0f, "Error: ${e.message}")
            return null
        }
    }

    private fun inferSingleModel(
        bitmap: Bitmap,
        progressCallback: ((Float, String) -> Unit)?
    ): StreamingResult? {
        val startTime = System.currentTimeMillis()
        try {
            val modelFile = findFile(MODEL_FILENAME) ?: run {
                Log.e(TAG, "Model file not found")
                return null
            }

            progressCallback?.invoke(0.05f, "Preprocessing image...")
            val scaledBitmap = SharpImagePreprocessor.resizeForSharp(bitmap)
            val imageData = preprocessImage(scaledBitmap)
            scaledBitmap.recycle()
            Log.d(TAG, "Preprocess: ${System.currentTimeMillis() - startTime}ms")

            progressCallback?.invoke(0.10f, "Loading INT8 model...")
            val loadStart = System.currentTimeMillis()
            val module = Module.load(modelFile.absolutePath, Module.LOAD_MODE_MMAP)
            Log.d(TAG, "Module.load: ${System.currentTimeMillis() - loadStart}ms. Memory: ${getMemoryInfo()}")

            progressCallback?.invoke(0.15f, "Running INT8 inference (single model)...")
            val inputTensor = Tensor.fromBlob(
                imageData,
                longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
            )
            val inputEValue = EValue.from(inputTensor)

            Log.d(TAG, "INT8 forward start. Memory: ${getMemoryInfo()}")
            val forwardStart = System.currentTimeMillis()
            val outputEValues = module.forward(inputEValue)
            val forwardTime = System.currentTimeMillis() - forwardStart
            Log.d(TAG, "INT8 forward done: ${forwardTime}ms. Memory: ${getMemoryInfo()}")

            val outputTensor = outputEValues[0].toTensor()
            val outputShape = outputTensor.shape()
            Log.d(TAG, "Output shape: ${outputShape.contentToString()}")

            val gaussianCount = if (outputShape.size == 2) {
                outputShape[0].toInt()
            } else {
                (outputTensor.numel() / PARAMS_PER_GAUSSIAN).toInt()
            }
            Log.d(TAG, "Gaussians: $gaussianCount")

            progressCallback?.invoke(0.70f, "Writing PLY ($gaussianCount Gaussians)...")
            val outputData = outputTensor.dataAsFloatArray
            val outputBuffer = FloatBuffer.wrap(outputData)

            module.destroy()
            System.gc()

            val result = writeStreamingPly(gaussianCount, outputBuffer, progressCallback)

            val totalTime = System.currentTimeMillis() - startTime
            Log.d(TAG, "INT8 single-model COMPLETE: ${totalTime}ms (forward=${forwardTime}ms)")
            Log.d(TAG, "  Gaussians: ${result.gaussianCount}, Room: ${result.roomWidth}m x ${result.roomHeight}m x ${result.roomDepth}m")

            progressCallback?.invoke(1.0f, "Done!")
            return result

        } catch (e: Exception) {
            Log.e(TAG, "INT8 inference failed: ${e.message}", e)
            progressCallback?.invoke(0f, "Error: ${e.message}")
            return null
        }
    }

    private fun preprocessImage(bitmap: Bitmap): FloatArray {
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
        return channelFloats
    }

    private fun writeStreamingPly(
        gaussianCount: Int,
        params: FloatBuffer,
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
            headerBuffer.put(headerBytes); headerBuffer.flip()
            channel.write(headerBuffer)

            val batchBuffer = plyBatch
            batchBuffer.clear()
            val scaleBoost = 1.3f; val minScale = 0.001f
            val lutScale = (LOGIT_LUT_SIZE - 1).toFloat()
            val invSHC0 = 1f / SH_C0
            val progressEvery = max(1, gaussianCount / 10)
            var processed = 0

            while (processed < gaussianCount) {
                val currentBatch = minOf(PLY_BATCH_SIZE, gaussianCount - processed)
                for (j in 0 until currentBatch) {
                    val offset = (processed + j) * PARAMS_PER_GAUSSIAN
                    val x = params.get(offset)
                    val y = -params.get(offset + 1)
                    val z = -params.get(offset + 2)
                    if (x < minX) minX = x; if (x > maxX) maxX = x
                    if (y < minY) minY = y; if (y > maxY) maxY = y
                    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z

                    batchBuffer.putFloat(x); batchBuffer.putFloat(y); batchBuffer.putFloat(z)
                    batchBuffer.putFloat(0f); batchBuffer.putFloat(0f); batchBuffer.putFloat(0f)

                    val r = params.get(offset + 11).coerceIn(0f, 1f)
                    val g = params.get(offset + 12).coerceIn(0f, 1f)
                    val b = params.get(offset + 13).coerceIn(0f, 1f)
                    batchBuffer.putFloat((r - 0.5f) * invSHC0)
                    batchBuffer.putFloat((g - 0.5f) * invSHC0)
                    batchBuffer.putFloat((b - 0.5f) * invSHC0)
                    zeroSHBuffer.clear()
                    batchBuffer.put(zeroSHBuffer)

                    val rawOpacity = params.get(offset + 3).coerceIn(0f, 1f)
                    batchBuffer.putFloat(LOGIT_LUT[(rawOpacity * lutScale).toInt().coerceIn(0, LOGIT_LUT_SIZE - 1)])

                    batchBuffer.putFloat(lnLut(max(params.get(offset + 4) * scaleBoost, minScale)))
                    batchBuffer.putFloat(lnLut(max(params.get(offset + 5) * scaleBoost, minScale)))
                    batchBuffer.putFloat(lnLut(max(params.get(offset + 6) * scaleBoost, minScale)))

                    val rw = params.get(offset + 7); val rx = params.get(offset + 8)
                    val ry = params.get(offset + 9); val rz = params.get(offset + 10)
                    val mag = sqrt(rw * rw + rx * rx + ry * ry + rz * rz)
                    val invMag = if (mag > 1e-8f) 1f / mag else 1f
                    batchBuffer.putFloat(rw * invMag); batchBuffer.putFloat(rx * invMag)
                    batchBuffer.putFloat(ry * invMag); batchBuffer.putFloat(rz * invMag)
                }
                batchBuffer.flip()
                batchBuffer.limit(currentBatch * BYTES_PER_VERTEX)
                while (batchBuffer.hasRemaining()) channel.write(batchBuffer)
                batchBuffer.clear()
                processed += currentBatch
                if (processed % progressEvery == 0 || processed == gaussianCount) {
                    progressCallback?.invoke(
                        0.70f + (processed.toFloat() / gaussianCount) * 0.25f,
                        "Writing PLY ($processed/$gaussianCount)..."
                    )
                }
            }
        }

        try {
            android.system.Os.link(plyFile.absolutePath, classicPlyFile.absolutePath)
        } catch (_: Exception) {
            plyFile.copyTo(classicPlyFile, overwrite = true)
        }

        Log.d(TAG, "PLY written: ${plyFile.absolutePath} (${plyFile.length()} bytes)")
        Log.d(TAG, "Room bounds: ${maxX - minX}m x ${maxY - minY}m x ${maxZ - minZ}m")

        return StreamingResult(
            plyFile = plyFile, classicPlyFile = classicPlyFile,
            gaussianCount = gaussianCount,
            roomWidth = maxX - minX, roomHeight = maxY - minY, roomDepth = maxZ - minZ
        )
    }

    private fun buildPlyHeader(gaussianCount: Int): String = buildString {
        append("ply\n")
        append("format binary_little_endian 1.0\n")
        append("element vertex $gaussianCount\n")
        append("property float x\nproperty float y\nproperty float z\n")
        append("property float nx\nproperty float ny\nproperty float nz\n")
        for (i in 0 until 3) append("property float f_dc_$i\n")
        for (i in 0 until 45) append("property float f_rest_$i\n")
        append("property float opacity\n")
        append("property float scale_0\nproperty float scale_1\nproperty float scale_2\n")
        append("property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n")
        append("end_header\n")
    }

    private fun findFile(filename: String): File? {
        val internal = File(internalModelsDir, filename)
        if (internal.exists() && internal.length() > 0) return internal
        val external = File(modelsDir, filename)
        if (external.exists() && external.length() > 0) return external
        return null
    }

    private fun getMemoryInfo(): String {
        val runtime = Runtime.getRuntime()
        val usedMB = (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024
        val maxMB = runtime.maxMemory() / 1024 / 1024
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val memInfo = android.app.ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        val availSysMB = memInfo.availMem / 1024 / 1024
        return "JVM: ${usedMB}/${maxMB}MB, SysAvail: ${availSysMB}MB"
    }

    fun release() {
        isInitialized = false
        Log.d(TAG, "ExecutorchInt8Sharp released")
    }
}
