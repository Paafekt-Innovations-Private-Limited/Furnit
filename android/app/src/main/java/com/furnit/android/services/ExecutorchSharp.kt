package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import com.furnit.android.utils.LogUtil
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileInputStream
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
 * SHARP Gaussian Splatting using ExecuTorch component-mode inference.
 *
 * Uses sliding pyramid patches to avoid OOM:
 * 1. Load sharp_single_patch.pte (~1.1GB) - processes one 384x384 patch at a time
 * 2. Extract 35 patches (25 at 1x, 9 at 0.5x, 1 at 0.25x)
 * 3. Merge encoded features on CPU into [1024, 96, 96]
 * 4. Load sharp_gaussian_head.pte (~7MB) - lightweight decoder
 * 5. Produce [14, 384, 384] Gaussian parameters → 147,456 Gaussians
 */
class ExecutorchSharp private constructor(private val context: Context) {

    companion object {
        private const val TAG = "ExecutorchSharp"
        private const val IMAGE_SIZE = 1536
        private const val PATCH_SIZE = 384
        private const val FEATURE_DIM = 1024
        private const val SPATIAL_SIZE = 24 // 384/16
        private const val GAUSSIAN_CHANNELS = 14
        private const val OUTPUT_SPATIAL = 384

        // Sliding pyramid configuration
        private const val GRID_1X = 5
        private const val GRID_05X = 3
        private const val PATCHES_1X = 25  // 5x5
        private const val PATCHES_05X = 9  // 3x3
        private const val PATCHES_025X = 1
        private const val TOTAL_PATCHES = 35

        // Merge overlap padding (matching NCNN component mode)
        private const val PADDING_1X = 3
        private const val PADDING_05X = 6

        private const val PARAMS_PER_GAUSSIAN = 14
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

        // 4-part split models (preferred - correct output, fits in memory)
        private const val SPLIT_PART1 = "sharp_split_part1.pte"  // Patch Encoder A ~582MB
        private const val SPLIT_PART2 = "sharp_split_part2.pte"  // Patch Encoder B ~577MB
        private const val SPLIT_PART3 = "sharp_split_part3.pte"  // Image Encoder A ~582MB
        private const val SPLIT_PART4 = "sharp_split_part4.pte"  // Decoder+Gaussians ~755MB
        private val SPLIT_FILENAMES = arrayOf(SPLIT_PART1, SPLIT_PART2, SPLIT_PART3, SPLIT_PART4)

        // Chunked Part 4: ViT on token slices (512 + 65) then decoder once. Lowers peak RAM.
        private const val SPLIT_PART4A_CHUNK_512 = "sharp_split_part4a_chunk_512.pte"
        private const val SPLIT_PART4A_CHUNK_65 = "sharp_split_part4a_chunk_65.pte"
        private const val SPLIT_PART4B = "sharp_split_part4b.pte"
        private const val IMAGE_TOKENS_SEQ_LEN = 577
        private const val CHUNK_LEN_FIRST = 512
        private const val CHUNK_LEN_LAST = IMAGE_TOKENS_SEQ_LEN - CHUNK_LEN_FIRST  // 65

        // Full pipeline models (single file). Prefer memory-optimized to reduce peak RAM.
        private val FULL_MODEL_FILENAMES = arrayOf(
            "sharp_full_memory_optimized.pte",  // Greedy planning + FP16 + XNNPACK
            "sharp_full_fp16.pte",
            "sharp_full_fp32.pte",
        )

        // Legacy component models (fallback - INCORRECT output from 7MB head)
        private val PATCH_ENCODER_FILENAMES = arrayOf(
            "sharp_single_patch_hybrid_standalone.pte",  // INT8 ~275MB, Vulkan+XNNPACK
            "sharp_single_patch_hybrid.pte",             // INT8 ~275MB, with .ptd separation
            "sharp_single_patch_xnnpack.pte",            // FP32 ~1.1GB, XNNPACK only
            "sharp_single_patch.pte",                    // FP32 ~1.1GB, portable fallback
        )
        private const val GAUSSIAN_HEAD_FILENAME = "sharp_gaussian_head.pte"

        private val EXTRA_SEARCH_DIRS = arrayOf(
            "/data/local/tmp/furnit/",
        )

        @Volatile
        private var instance: ExecutorchSharp? = null

        /** Single mutex for init/preload/inference — prevents overlap and partial copy races. */
        private val mutex = Mutex()

        fun getInstance(context: Context): ExecutorchSharp {
            return instance ?: synchronized(this) {
                instance ?: ExecutorchSharp(context.applicationContext).also {
                    instance = it
                    LogUtil.d(TAG, "ExecutorchSharp singleton created")
                }
            }
        }
    }

    @Volatile
    private var isInitialized = false

    /** Internal storage (fast mmap, direct ext4). Primary location for model files. */
    private val internalModelsDir: File by lazy {
        File(context.filesDir, "models").also { it.mkdirs() }
    }

    /** External storage (slow FUSE). Fallback; models copied to internal when found. */
    private val externalModelsDir: File? get() = context.getExternalFilesDir("models")

    /** ONNX-style: temp dir for intermediate tensors between parts (one part at a time, unload between). */
    private val executorchTempDir: File by lazy {
        File(context.cacheDir, "executorch_sharp_temp").also { it.mkdirs() }
    }

    private val zeroSHBuffer: ByteBuffer by lazy {
        ByteBuffer.allocateDirect(45 * 4).apply { order(ByteOrder.LITTLE_ENDIAN) }
    }
    private val plyBatch: ByteBuffer by lazy {
        ByteBuffer.allocateDirect(BYTES_PER_VERTEX * PLY_BATCH_SIZE).apply { order(ByteOrder.LITTLE_ENDIAN) }
    }

    /** Save FloatArray + shape to file (same format as SplitOnnxSharp: 4 bytes ndim, 8*ndim shape, then floats). */
    private fun saveFloatArrayToFile(file: File, data: FloatArray, shape: LongArray) {
        RandomAccessFile(file, "rw").use { raf ->
            val channel = raf.channel
            val header = ByteBuffer.allocate(4 + 8 * shape.size).order(ByteOrder.LITTLE_ENDIAN)
            header.putInt(shape.size)
            shape.forEach { header.putLong(it) }
            header.flip()
            channel.write(header)
            val dataBuf = ByteBuffer.allocate(data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            dataBuf.asFloatBuffer().put(data)
            channel.write(dataBuf)
        }
        LogUtil.d(TAG, "Saved ${file.name} shape=${shape.contentToString()} size=${data.size}")
    }

    /** Load FloatArray and shape from file. Caller must use shape for Tensor.fromBlob. */
    private fun loadFloatArrayFromFile(file: File): Pair<FloatArray, LongArray> {
        RandomAccessFile(file, "r").use { raf ->
            val channel = raf.channel
            val header = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
            channel.read(header)
            header.flip()
            val numDims = header.int
            val shapeBuf = ByteBuffer.allocate(8 * numDims).order(ByteOrder.LITTLE_ENDIAN)
            channel.read(shapeBuf)
            shapeBuf.flip()
            val shape = LongArray(numDims) { shapeBuf.long }
            val totalFloats = shape.fold(1L) { acc, d -> acc * d }.toInt()
            val data = FloatArray(totalFloats)
            val dataBuf = channel.map(FileChannel.MapMode.READ_ONLY, 4L + 8 * numDims, totalFloats * 4L).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            dataBuf.get(data)
            LogUtil.d(TAG, "Loaded ${file.name} shape=${shape.contentToString()}")
            return data to shape
        }
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
        LogUtil.d(TAG, "findFile: looking for $filename")
        val internal = File(internalModelsDir, filename)
        if (internal.exists() && internal.length() > 0) {
            LogUtil.d(TAG, "findFile: $filename found internal path=${internal.absolutePath} size=${internal.length()}")
            return internal
        }
        externalModelsDir?.let { ext ->
            val external = File(ext, filename)
            if (external.exists() && external.length() > 0) {
                LogUtil.d(TAG, "findFile: $filename found external path=${external.absolutePath}")
                return external
            }
        }
        for (dir in EXTRA_SEARCH_DIRS) {
            val file = File(dir, filename)
            if (file.exists() && file.length() > 0) {
                LogUtil.d(TAG, "findFile: $filename found extra path=${file.absolutePath}")
                return file
            }
        }
        LogUtil.w(TAG, "findFile: $filename NOT FOUND")
        return null
    }

    private fun getMemoryInfo(): String {
        return try {
            val runtime = Runtime.getRuntime()
            val usedMB = (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024
            val maxMB = runtime.maxMemory() / 1024 / 1024
            val availSysMB = getAvailSysMemMB()
            "JVM: ${usedMB}/${maxMB}MB, SysAvail: ${availSysMB}MB"
        } catch (e: Exception) {
            "JVM: ${(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024 / 1024}MB"
        }
    }

    /** System available memory in MB; -1 if unknown. Part 4 needs ~2GB+ headroom to avoid LMK. */
    private fun getAvailSysMemMB(): Long {
        return try {
            val activityManager = context.getSystemService(android.content.Context.ACTIVITY_SERVICE) as? android.app.ActivityManager
            activityManager?.let { am ->
                val memInfo = android.app.ActivityManager.MemoryInfo()
                am.getMemoryInfo(memInfo)
                memInfo.availMem / 1024 / 1024
            } ?: -1L
        } catch (_: Exception) { -1L }
    }

    /** Resolve source file for a split model (external or extra dirs). Internal is not a source. */
    private fun findSourceForCopy(filename: String): File? {
        externalModelsDir?.let { ext ->
            val f = File(ext, filename)
            if (f.exists() && f.length() > 0) return f
        }
        for (dir in EXTRA_SEARCH_DIRS) {
            val f = File(dir, filename)
            if (f.exists() && f.length() > 0) return f
        }
        return null
    }

    /** True iff all 4 split models exist in internal storage with valid size. */
    private fun areAllSplitModelsInternal(): Boolean {
        for (filename in SPLIT_FILENAMES) {
            val f = File(internalModelsDir, filename)
            val src = findSourceForCopy(filename)
            if (src == null) {
                if (!f.exists() || f.length() <= 0) return false
            } else {
                if (!f.exists() || f.length() != src.length()) return false
            }
        }
        return true
    }

    /** True iff chunked Part 4 models exist (4a_chunk_512, 4a_chunk_65, 4b). Use for lower peak RAM. */
    private fun isChunkedPart4Available(): Boolean {
        return findFile(SPLIT_PART4A_CHUNK_512) != null &&
            findFile(SPLIT_PART4A_CHUNK_65) != null &&
            findFile(SPLIT_PART4B) != null
    }

    /** Copy split models from external/extra to internal storage for faster mmap.
     * Uses atomic copy (tmp + fsync + rename). Blocks inference until all 4 are internal-ready. */
    private fun ensureSplitModelsInInternalStorage(): Boolean {
        for (filename in SPLIT_FILENAMES) {
            val dst = File(internalModelsDir, filename)
            val src = findSourceForCopy(filename)
            if (src == null) {
                if (!dst.exists() || dst.length() <= 0) {
                    LogUtil.w(TAG, "ensureSplitModels: no source for $filename")
                    return false
                }
                continue
            }
            if (dst.exists() && dst.length() == src.length()) {
                LogUtil.d(TAG, "ensureSplitModels: $filename already internal (${dst.length() / 1024 / 1024}MB)")
                continue
            }
            try {
                LogUtil.d(TAG, "Copying $filename (${src.length() / 1024 / 1024} MB) to internal...")
                val tmp = File(internalModelsDir, "$filename.tmp")
                FileInputStream(src).use { fis ->
                    FileOutputStream(tmp).use { fos ->
                        val buf = ByteArray(1024 * 1024)
                        var total = 0L
                        while (true) {
                            val n = fis.read(buf)
                            if (n <= 0) break
                            fos.write(buf, 0, n)
                            total += n
                        }
                        fos.fd.sync()
                    }
                }
                if (tmp.length() != src.length()) {
                    LogUtil.e(TAG, "Copy validation failed: $filename tmp=${tmp.length()} src=${src.length()}")
                    tmp.delete()
                    return false
                }
                if (!tmp.renameTo(dst)) {
                    LogUtil.e(TAG, "Copy rename failed: $filename")
                    tmp.delete()
                    return false
                }
                LogUtil.d(TAG, "Copied $filename to ${dst.absolutePath} size=${dst.length()}")
            } catch (e: Exception) {
                LogUtil.e(TAG, "Failed to copy $filename: ${e.message}")
                File(internalModelsDir, "$filename.tmp").delete()
                return false
            }
        }
        return true
    }

    private var useSplitMode = false
    private var useFullModel = false

    private fun isSplitModelReady(): Boolean {
        val ready = SPLIT_FILENAMES.all { findFile(it) != null }
        LogUtil.d(TAG, "isSplitModelReady: $ready")
        return ready
    }

    private fun findFullModel(): File? {
        for (filename in FULL_MODEL_FILENAMES) {
            val file = findFile(filename)
            if (file != null) return file
        }
        return null
    }

    private fun findPatchEncoder(): File? {
        for (filename in PATCH_ENCODER_FILENAMES) {
            val file = findFile(filename)
            if (file != null) return file
        }
        return null
    }

    fun isModelReady(): Boolean {
        // Disabled: XNNPACK models cause SIGSEGV. Use ExecutorchInt8Sharp (Vulkan-only) instead.
        return false
    }

    /**
     * Preload Part1 encoder: load + destroy (fast validation). NO warmup forward.
     * Warmup was removed: on CPU fallback it took 10+ min; with XNNPACK/Vulkan models
     * the first forward is already acceptable. Re-enabling warmup would block screen open.
     */
    suspend fun preloadAndWarmup(progress: ((String) -> Unit)? = null) = withContext(Dispatchers.IO) {
        LogUtil.d(TAG, "preloadAndWarmup ENTER (load-only, no warmup)")
        mutex.withLock {
            if (!isInitialized) initializeImpl()
        }
        if (!useSplitMode) {
            LogUtil.d(TAG, "preloadAndWarmup: not split mode, skipping")
            return@withContext
        }
        val part1File = findFile(SPLIT_PART1) ?: return@withContext
        LogUtil.d(TAG, "preloadAndWarmup: validating Part1 from ${part1File.absolutePath} size=${part1File.length() / 1024 / 1024}MB")
        progress?.invoke("Validating encoder A...")
        val t0 = System.currentTimeMillis()
        val module = Module.load(part1File.absolutePath, Module.LOAD_MODE_MMAP)
        LogUtil.d(TAG, "PRELOAD Part1 load ${System.currentTimeMillis() - t0}ms (no warmup forward)")
        module.destroy()
        System.gc()
        LogUtil.d(TAG, "preloadAndWarmup DONE total=${System.currentTimeMillis() - t0}ms")
        progress?.invoke("Preload done")
    }

    /** Caller must hold mutex. Prefer split over full (split uses less peak RAM; Part 4 can use greedy-planned .pte). */
    private fun initializeImpl(): Boolean {
        LogUtil.d(TAG, "initialize ENTER. Memory: ${getMemoryInfo()}")

        // Copy split models to internal storage (required before inference)
        if (!ensureSplitModelsInInternalStorage()) {
            LogUtil.e(TAG, "initialize: copy to internal failed")
        }
        // Priority 1: 4-part split (lower peak RAM, Part 4 exported with greedy memory planning)
        if (areAllSplitModelsInternal()) {
            for (fn in SPLIT_FILENAMES) {
                val f = File(internalModelsDir, fn)
                LogUtil.d(TAG, "  Split internal: ${f.name} (${f.length() / 1024 / 1024}MB)")
            }
            LogUtil.d(TAG, "Split model (preferred): 4 parts")
            useSplitMode = true
            useFullModel = false
            isInitialized = true
            return true
        }

        // Priority 2: Full model
        val fullModel = findFullModel()
        if (fullModel != null) {
            LogUtil.d(TAG, "Full model: ${fullModel.name} (${fullModel.length() / 1024 / 1024}MB)")
            useSplitMode = false
            useFullModel = true
            isInitialized = true
            return true
        }

        // Priority 3: Legacy component (wrong output)
        val patchFile = findPatchEncoder()
        val headFile = findFile(GAUSSIAN_HEAD_FILENAME)
        if (patchFile != null && headFile != null) {
            LogUtil.w(TAG, "Using LEGACY component mode (7MB head - incorrect output)")
            useSplitMode = false
            useFullModel = false
            isInitialized = true
            return true
        }

        LogUtil.e(TAG, "No ExecuTorch models found")
        return false
    }

    fun initialize(): Boolean = runBlocking { mutex.withLock { initializeImpl() } }

    /** Native heap allocated (MB); -1 if unknown. Use to verify destroy() frees. */
    private fun nativeHeapMB(): Long = try {
        android.os.Debug.getNativeHeapAllocatedSize() / 1024 / 1024
    } catch (_: Exception) { -1L }

    /**
     * Run component-mode SHARP inference.
     * Fix 1: Entire inference under lock to prevent concurrent overlap (double-tap, recomposition).
     */
    suspend fun inferStreaming(
        bitmap: Bitmap,
        progressCallback: ((Float, String) -> Unit)? = null
    ): StreamingResult? = withContext(Dispatchers.IO) {
        mutex.withLock {
            if (!isInitialized) {
                LogUtil.e(TAG, "Not initialized")
                return@withContext null
            }
            val streamStartTime = System.currentTimeMillis()
            LogUtil.d(TAG, "inferStreaming ENTER bitmap=${bitmap.width}x${bitmap.height} thread=${Thread.currentThread().name} split=$useSplitMode full=$useFullModel Memory: ${getMemoryInfo()} native=${nativeHeapMB()}MB")

            try {
                when {
                    useSplitMode -> {
                        LogUtil.d(TAG, "inferStreaming calling inferSplitMode()")
                        return@withContext inferSplitMode(bitmap, progressCallback)
                    }
                    useFullModel -> {
                        LogUtil.d(TAG, "inferStreaming calling inferFullModel()")
                        return@withContext inferFullModel(bitmap, progressCallback)
                    }
                    else -> LogUtil.d(TAG, "inferStreaming using legacy component mode")
                }

            val startTime = System.currentTimeMillis()

            // Step 1: Scale input image to 1536x1536
            progressCallback?.invoke(0.02f, "Preprocessing image...")
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
            LogUtil.d(TAG, "Image scaled to ${IMAGE_SIZE}x${IMAGE_SIZE}")

            // Step 2: Create downsampled images for multi-scale pyramid
            val halfSize = IMAGE_SIZE / 2  // 768
            val quarterBitmap = Bitmap.createScaledBitmap(scaledBitmap, PATCH_SIZE, PATCH_SIZE, true)
            val halfBitmap = Bitmap.createScaledBitmap(scaledBitmap, halfSize, halfSize, true)

            // Step 3: Load patch encoder
            progressCallback?.invoke(0.05f, "Loading patch encoder...")
            val patchEncoderFile = findPatchEncoder()!!
            val encoderLoadStart = System.currentTimeMillis()
            val patchEncoder = Module.load(patchEncoderFile.absolutePath, Module.LOAD_MODE_MMAP)
            LogUtil.d(TAG, "Patch encoder loaded in ${System.currentTimeMillis() - encoderLoadStart}ms")

            // Step 4: Encode all 35 patches (25 @ 1x + 9 @ 0.5x + 1 @ 0.25x)
            val patchFeatures1x = ArrayList<FloatArray>(PATCHES_1X)
            val patchFeatures05x = ArrayList<FloatArray>(PATCHES_05X)
            var patchCount = 0
            val encodeStart = System.currentTimeMillis()

            // 1x patches (5x5 grid)
            val stride1x = (IMAGE_SIZE - PATCH_SIZE) / (GRID_1X - 1) // 288
            for (i in 0 until GRID_1X) {
                for (j in 0 until GRID_1X) {
                    val y = i * stride1x
                    val x = j * stride1x
                    val patchBitmap = Bitmap.createBitmap(scaledBitmap, x, y, PATCH_SIZE, PATCH_SIZE)
                    patchFeatures1x.add(encodePatch(patchEncoder, patchBitmap))
                    patchBitmap.recycle()
                    patchCount++
                    if (patchCount % 5 == 0) {
                        val progress = 0.05f + (patchCount.toFloat() / TOTAL_PATCHES) * 0.50f
                        progressCallback?.invoke(progress, "Encoding patch $patchCount/$TOTAL_PATCHES...")
                    }
                }
            }
            LogUtil.d(TAG, "1x patches encoded: ${patchFeatures1x.size} in ${System.currentTimeMillis() - encodeStart}ms")

            // 0.5x patches (3x3 grid)
            val stride05x = (halfSize - PATCH_SIZE) / (GRID_05X - 1) // 192
            val encode05Start = System.currentTimeMillis()
            for (i in 0 until GRID_05X) {
                for (j in 0 until GRID_05X) {
                    val y = i * stride05x
                    val x = j * stride05x
                    val patchBitmap = Bitmap.createBitmap(halfBitmap, x, y, PATCH_SIZE, PATCH_SIZE)
                    patchFeatures05x.add(encodePatch(patchEncoder, patchBitmap))
                    patchBitmap.recycle()
                    patchCount++
                }
            }
            LogUtil.d(TAG, "0.5x patches encoded: ${patchFeatures05x.size} in ${System.currentTimeMillis() - encode05Start}ms")
            halfBitmap.recycle()

            // 0.25x patch (single 384x384)
            val encode025Start = System.currentTimeMillis()
            val features025x = encodePatch(patchEncoder, quarterBitmap)
            quarterBitmap.recycle()
            patchCount++
            LogUtil.d(TAG, "0.25x patch encoded in ${System.currentTimeMillis() - encode025Start}ms")

            val totalEncodeTime = System.currentTimeMillis() - encodeStart
            LogUtil.d(TAG, "All $patchCount patches encoded in ${totalEncodeTime}ms (${totalEncodeTime / patchCount}ms/patch)")
            progressCallback?.invoke(0.55f, "All $patchCount patches encoded")

            scaledBitmap.recycle()

            // Step 5: Unload patch encoder to free memory
            patchEncoder.destroy()
            LogUtil.d(TAG, "Patch encoder released")
            System.gc()

            // Step 6: Merge features on CPU
            progressCallback?.invoke(0.58f, "Merging features...")
            val mergeStart = System.currentTimeMillis()
            val merged1x = mergePatchGrid(patchFeatures1x, GRID_1X, PADDING_1X)
            patchFeatures1x.clear()
            LogUtil.d(TAG, "Merged 1x: ${merged1x.size} floats")

            // Merge 0.5x and 0.25x are not used by current gaussian head
            // but we encode them for future use / quality improvement
            patchFeatures05x.clear()
            LogUtil.d(TAG, "Merge done in ${System.currentTimeMillis() - mergeStart}ms")

            // Step 7: Load gaussian head
            progressCallback?.invoke(0.62f, "Loading Gaussian head...")
            val headFile = findFile(GAUSSIAN_HEAD_FILENAME)!!
            val headLoadStart = System.currentTimeMillis()
            val gaussianHead = Module.load(headFile.absolutePath, Module.LOAD_MODE_MMAP)
            LogUtil.d(TAG, "Gaussian head loaded in ${System.currentTimeMillis() - headLoadStart}ms")

            // Step 8: Run gaussian head on merged features
            progressCallback?.invoke(0.65f, "Running Gaussian head...")
            val mergedSize = getMergedSize(GRID_1X, SPATIAL_SIZE, PADDING_1X)
            val mergedTensor = Tensor.fromBlob(
                merged1x,
                longArrayOf(1, FEATURE_DIM.toLong(), mergedSize.toLong(), mergedSize.toLong())
            )

            val headStart = System.currentTimeMillis()
            val headOutputs = gaussianHead.forward(EValue.from(mergedTensor))
            val headTime = System.currentTimeMillis() - headStart
            LogUtil.d(TAG, "Gaussian head completed in ${headTime}ms")

            val headOutput = headOutputs[0].toTensor().getDataAsFloatArray()

            // Step 9: Release gaussian head
            gaussianHead.destroy()
            LogUtil.d(TAG, "Gaussian head released")

            // Step 10: Extract Gaussians from [1, 14, 384, 384] output
            progressCallback?.invoke(0.70f, "Extracting Gaussians...")
            val gaussianCount = OUTPUT_SPATIAL * OUTPUT_SPATIAL // 147,456
            val params = extractGaussians(headOutput, OUTPUT_SPATIAL, OUTPUT_SPATIAL)
            LogUtil.d(TAG, "Extracted $gaussianCount Gaussians")

            // Step 11: Write PLY
            progressCallback?.invoke(0.75f, "Writing PLY ($gaussianCount Gaussians)...")
            val result = writeStreamingPlyPacked(gaussianCount, params, progressCallback)

            val elapsed = System.currentTimeMillis() - startTime
            LogUtil.d(TAG, "ExecuTorch component-mode SHARP completed: $gaussianCount Gaussians in ${elapsed}ms")

            progressCallback?.invoke(1.0f, "Done!")
            return@withContext result

        } catch (e: Exception) {
            LogUtil.e(TAG, "ExecuTorch SHARP inference failed after ${System.currentTimeMillis() - streamStartTime}ms native=${nativeHeapMB()}MB", e)
            progressCallback?.invoke(0f, "Error: ${e.message}")
            return@withContext null
        }
        }
    }

    /**
     * 4-part split inference: same pipeline as ONNX Split / LiteRT Split.
     * Part 1+2 run 35 times (per patch), Part 3+4 run once.
     * Produces correct 1,179,648 Gaussians.
     */
    /**
     * 4-part split inference with REAL memory control:
     * - Part 1: run patch encoder A per patch
     *   - store ONLY tokens for Part 2
     *   - merge latent0/latent1 (block5/tokens spatial) ON THE FLY for 1x patches
     * - Part 2: run patch encoder B per token set
     *   - merge x0Feat (1x) and x1Feat (0.5x) ON THE FLY
     *   - keep only x2Feat for 0.25x
     * - Part 3/4 unchanged
     */
    private suspend fun inferSplitMode(
        bitmap: Bitmap,
        progressCallback: ((Float, String) -> Unit)?
    ): StreamingResult? {

        /** When [blendCount] is non-null, accumulate into [output] and increment [blendCount] per spatial pixel (ONNX-style blend in overlap). Otherwise overwrite. */
        fun mergeOnePatchInto(
            output: FloatArray,
            outSize: Int,
            patch: FloatArray,
            gridI: Int,
            gridJ: Int,
            gridSize: Int,
            padding: Int,
            blendCount: IntArray? = null
        ) {
            val patchH = SPATIAL_SIZE
            val patchW = SPATIAL_SIZE
            val C = FEATURE_DIM

            val srcY0 = if (gridJ == 0) 0 else padding
            val srcY1 = if (gridJ == gridSize - 1) patchH else (patchH - padding)
            val copyH = srcY1 - srcY0

            val srcX0 = if (gridI == 0) 0 else padding
            val srcX1 = if (gridI == gridSize - 1) patchW else (patchW - padding)
            val copyW = srcX1 - srcX0

            // Correct placement math for overlapped tiling:
            // first patch contributes (patchW - padding), inner patches (patchW - 2*padding)
            val firstContrib = patchW - padding
            val innerContrib = patchW - 2 * padding
            val outX = if (gridI == 0) 0 else firstContrib + (gridI - 1) * innerContrib
            val outY = if (gridJ == 0) 0 else firstContrib + (gridJ - 1) * innerContrib

            if (outX < 0 || outY < 0 || outX + copyW > outSize || outY + copyH > outSize) {
                LogUtil.e(TAG, "mergeOnePatchInto OOB: grid=($gridI,$gridJ)/$gridSize pad=$padding " +
                        "outSize=$outSize out=($outX,$outY) copy=($copyW,$copyH) srcX=[$srcX0,$srcX1) srcY=[$srcY0,$srcY1)")
                throw IllegalStateException("mergeOnePatchInto bounds invalid")
            }

            val patchHW = patchH * patchW
            val outHW = outSize * outSize
            val accumulating = blendCount != null

            for (c in 0 until C) {
                val srcBase = c * patchHW
                val dstBase = c * outHW
                for (dy in 0 until copyH) {
                    val outRow = outY + dy
                    for (dx in 0 until copyW) {
                        val srcIdx = srcBase + (srcY0 + dy) * patchW + (srcX0 + dx)
                        val dstIdx = dstBase + outRow * outSize + (outX + dx)
                        if (accumulating) {
                            output[dstIdx] += patch[srcIdx]
                            if (c == 0) blendCount!![outRow * outSize + (outX + dx)]++
                        } else {
                            output[dstIdx] = patch[srcIdx]
                        }
                    }
                }
            }
        }

        /** Normalize merged feature maps by per-spatial contribution count (blend match to ONNX). */
        fun normalizeMergedByCount(output: FloatArray, outSize: Int, count: IntArray) {
            val outHW = outSize * outSize
            for (idx in 0 until outHW) {
                val n = maxOf(1, count[idx])
                val invN = 1f / n
                for (c in 0 until FEATURE_DIM) output[c * outHW + idx] *= invN
            }
        }

        // Reusable buffer to avoid allocating 25x big spatial arrays
        val tempSpatial = FloatArray(FEATURE_DIM * SPATIAL_SIZE * SPATIAL_SIZE)

        fun reshapeTokensToSpatialInto(tokens: FloatArray, spatialOut: FloatArray) {
            // tokens: [577,1024]  -> spatialOut: [1024,24,24]  (skip CLS token)
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

        val startTime = System.currentTimeMillis()

        try {
            LogUtil.d(TAG, "inferSplitMode ENTER bitmap=${bitmap.width}x${bitmap.height} thread=${Thread.currentThread().name} Memory: ${getMemoryInfo()}")

            progressCallback?.invoke(0.02f, "Preprocessing...")
            val tScale0 = System.currentTimeMillis()
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
            LogUtil.d(TAG, "inferSplitMode scale done in ${System.currentTimeMillis() - tScale0}ms")

            // Pyramid bitmaps for patch extraction (imageData deferred to before Part 3)
            val halfSize = IMAGE_SIZE / 2
            val halfBitmap = Bitmap.createScaledBitmap(scaledBitmap, halfSize, halfSize, true)
            val quarterBitmap = Bitmap.createScaledBitmap(scaledBitmap, PATCH_SIZE, PATCH_SIZE, true)

            // Pre-allocate merged outputs (fixed memory, no patch lists)
            val mergedSize1x = getMergedSize(GRID_1X, SPATIAL_SIZE, PADDING_1X)   // 96
            val mergedSize05x = getMergedSize(GRID_05X, SPATIAL_SIZE, PADDING_05X) // 48
            val stride1x = (IMAGE_SIZE - PATCH_SIZE) / 4 // 288
            val stride05x = (halfSize - PATCH_SIZE) / 2  // 192
            LogUtil.d(TAG, "inferSplitMode: mergedSize1x=$mergedSize1x mergedSize05x=$mergedSize05x stride1x=$stride1x stride05x=$stride05x")

            // Part 1+2 + save in a block so latent0/latent1/x0Feat/x1Feat/x2/imageData go out of scope after save (~180MB reclaimable)
            data class Part12Saved(val inputImageFile: File, val mergedSize1x: Int, val mergedSize05x: Int, val part1Time: Long, val part2Time: Long)
            val part12Saved = run {
            val latent0 = FloatArray(FEATURE_DIM * mergedSize1x * mergedSize1x) // from block5Spatial1x
            val latent1 = FloatArray(FEATURE_DIM * mergedSize1x * mergedSize1x) // from tokensSpatial1x
            val x0Feat  = FloatArray(FEATURE_DIM * mergedSize1x * mergedSize1x) // Part2 features 1x
            val x1Feat  = FloatArray(FEATURE_DIM * mergedSize05x * mergedSize05x) // Part2 features 0.5x
            var x2Feat: FloatArray? = null // Part2 features 0.25x
            val count1x = IntArray(mergedSize1x * mergedSize1x)   // blend count for ONNX-style merge
            val count05x = IntArray(mergedSize05x * mergedSize05x)

            // -------------------------
            // Part 1 + Part 2: Load both, stream per-patch (no allTokens ~78MB)
            // -------------------------
            progressCallback?.invoke(0.05f, "Loading encoder A + B...")
            val part1File = findFile(SPLIT_PART1) ?: run {
                LogUtil.e(TAG, "Missing ${SPLIT_PART1}")
                return null
            }
            val part2File = findFile(SPLIT_PART2) ?: run {
                LogUtil.e(TAG, "Missing ${SPLIT_PART2}")
                return null
            }
            LogUtil.d(TAG, "Part1 load from ${part1File.absolutePath} size=${part1File.length() / 1024 / 1024}MB")
            LogUtil.d(TAG, "Part2 load from ${part2File.absolutePath} size=${part2File.length() / 1024 / 1024}MB")
            val tLoad0 = System.currentTimeMillis()
val module1 = Module.load(part1File.absolutePath, Module.LOAD_MODE_MMAP)
        val module2 = Module.load(part2File.absolutePath, Module.LOAD_MODE_MMAP)
            LogUtil.d(TAG, "Part1+Part2 load done in ${System.currentTimeMillis() - tLoad0}ms")

            var patchCount = 0
            val part12Start = System.currentTimeMillis()

            // 1x patches (5x5): Part1->tokens+block5, merge latent0/latent1, Part2(tokens)->feat, merge x0Feat
            for (i in 0 until GRID_1X) {
                for (j in 0 until GRID_1X) {
                    val patchBitmap = Bitmap.createBitmap(
                        scaledBitmap,
                        j * stride1x,
                        i * stride1x,
                        PATCH_SIZE,
                        PATCH_SIZE
                    )
                    if (patchCount == 0) LogUtil.d(TAG, "P1 PATCH 0 preprocess start")
                    val patchData = preprocessPatch(patchBitmap)
                    if (patchCount == 0) LogUtil.d(TAG, "P1 PATCH 0 preprocess done")
                    patchBitmap.recycle()

                    val inputTensor = Tensor.fromBlob(
                        patchData,
                        longArrayOf(1, 3, PATCH_SIZE.toLong(), PATCH_SIZE.toLong())
                    )
                    if (patchCount == 0) LogUtil.d(TAG, "P1 PATCH 0 forward start")
                    val tForward = if (patchCount == 0) System.currentTimeMillis() else 0L
                    val out1 = module1.forward(EValue.from(inputTensor))
                    if (patchCount == 0) LogUtil.d(TAG, "P1 PATCH 0 forward done in ${System.currentTimeMillis() - tForward}ms")
                    val tokens = out1[0].toTensor().getDataAsFloatArray()
                    val block5 = out1[1].toTensor().getDataAsFloatArray()

                    reshapeTokensToSpatialInto(block5, tempSpatial)
                    mergeOnePatchInto(output = latent0, outSize = mergedSize1x, patch = tempSpatial, gridI = j, gridJ = i, gridSize = GRID_1X, padding = PADDING_1X, blendCount = count1x)
                    reshapeTokensToSpatialInto(tokens, tempSpatial)
                    mergeOnePatchInto(output = latent1, outSize = mergedSize1x, patch = tempSpatial, gridI = j, gridJ = i, gridSize = GRID_1X, padding = PADDING_1X, blendCount = count1x)

                    val feat = module2.forward(EValue.from(Tensor.fromBlob(tokens, longArrayOf(1, 577, 1024))))[0].toTensor().getDataAsFloatArray()
                    mergeOnePatchInto(output = x0Feat, outSize = mergedSize1x, patch = feat, gridI = j, gridJ = i, gridSize = GRID_1X, padding = PADDING_1X, blendCount = count1x)

                    patchCount++
                    if (patchCount % 5 == 0) {
                        progressCallback?.invoke(0.05f + (patchCount.toFloat() / TOTAL_PATCHES) * 0.40f, "Part 1+2: Patch $patchCount/$TOTAL_PATCHES...")
                    }
                }
            }
            normalizeMergedByCount(latent0, mergedSize1x, count1x)
            normalizeMergedByCount(latent1, mergedSize1x, count1x)
            normalizeMergedByCount(x0Feat, mergedSize1x, count1x)

            // 0.5x patches (3x3): Part1->tokens, Part2(tokens)->feat, merge x1Feat
            for (i in 0 until GRID_05X) {
                for (j in 0 until GRID_05X) {
                    val patchBitmap = Bitmap.createBitmap(halfBitmap, j * stride05x, i * stride05x, PATCH_SIZE, PATCH_SIZE)
                    val patchData = preprocessPatch(patchBitmap)
                    patchBitmap.recycle()
                    val inputTensor = Tensor.fromBlob(patchData, longArrayOf(1, 3, PATCH_SIZE.toLong(), PATCH_SIZE.toLong()))
                    val tokens = module1.forward(EValue.from(inputTensor))[0].toTensor().getDataAsFloatArray()
                    val feat = module2.forward(EValue.from(Tensor.fromBlob(tokens, longArrayOf(1, 577, 1024))))[0].toTensor().getDataAsFloatArray()
                    mergeOnePatchInto(output = x1Feat, outSize = mergedSize05x, patch = feat, gridI = j, gridJ = i, gridSize = GRID_05X, padding = PADDING_05X, blendCount = count05x)
                    patchCount++
                }
            }
            normalizeMergedByCount(x1Feat, mergedSize05x, count05x)
            halfBitmap.recycle()

            // 0.25x patch: Part1->tokens, Part2(tokens)->feat, x2Feat=feat
            val qData = preprocessPatch(quarterBitmap)
            quarterBitmap.recycle()
            val qTokens = module1.forward(EValue.from(Tensor.fromBlob(qData, longArrayOf(1, 3, PATCH_SIZE.toLong(), PATCH_SIZE.toLong()))))[0].toTensor().getDataAsFloatArray()
            val qFeat = module2.forward(EValue.from(Tensor.fromBlob(qTokens, longArrayOf(1, 577, 1024))))[0].toTensor().getDataAsFloatArray()
            x2Feat = qFeat
            patchCount++

            module1.destroy()
            module2.destroy()
            System.gc()
            System.runFinalization()
            LogUtil.d(TAG, "Part 1+2 done: $patchCount patches in ${System.currentTimeMillis() - part12Start}ms (streamed, no allTokens) Memory: ${getMemoryInfo()}")

            val x2 = x2Feat ?: run {
                LogUtil.e(TAG, "Missing x2Feat (0.25x) output")
                return null
            }

            // ONNX-style: save Part 1+2 intermediates to disk
            executorchTempDir.listFiles()?.forEach { it.delete() }
            saveFloatArrayToFile(File(executorchTempDir, "latent0.tensor"), latent0, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize1x.toLong(), mergedSize1x.toLong()))
            saveFloatArrayToFile(File(executorchTempDir, "latent1.tensor"), latent1, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize1x.toLong(), mergedSize1x.toLong()))
            saveFloatArrayToFile(File(executorchTempDir, "x0Feat.tensor"), x0Feat, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize1x.toLong(), mergedSize1x.toLong()))
            saveFloatArrayToFile(File(executorchTempDir, "x1Feat.tensor"), x1Feat, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize05x.toLong(), mergedSize05x.toLong()))
            saveFloatArrayToFile(File(executorchTempDir, "x2.tensor"), x2, longArrayOf(1, FEATURE_DIM.toLong(), SPATIAL_SIZE.toLong(), SPATIAL_SIZE.toLong()))

            val tPrep0 = System.currentTimeMillis()
            val imageData = preprocessPatch(scaledBitmap)
            val inputImageFile = File(executorchTempDir, "input_image.tensor")
            saveFloatArrayToFile(inputImageFile, imageData, longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong()))
            scaledBitmap.recycle()
            val part12Time = System.currentTimeMillis() - part12Start
            Part12Saved(inputImageFile, mergedSize1x, mergedSize05x, part12Time / 2, part12Time / 2)
            }
            System.gc()
            System.runFinalization()
            LogUtil.d(TAG, "Part 1+2 intermediates on disk, block closed (latent0/1 x0Feat/x1Feat/x2/imageData reclaimable). Memory: ${getMemoryInfo()}")

            val inputImageFile = part12Saved.inputImageFile

            // Part 3: load only Part 3, input from disk, run, save imageTokens, unload (block so imageDataP3/imageTokens reclaimable)
            progressCallback?.invoke(0.50f, "Part 3: Image encoder...")
            val part3File = findFile(SPLIT_PART3) ?: run {
                LogUtil.e(TAG, "Missing ${SPLIT_PART3}")
                return null
            }
            val part3Start = System.currentTimeMillis()
            run {
                val (imageDataP3, imageShape) = loadFloatArrayFromFile(inputImageFile)
                val module3 = Module.load(part3File.absolutePath, Module.LOAD_MODE_MMAP)
                val imageTokensOut = module3.forward(EValue.from(Tensor.fromBlob(imageDataP3, imageShape)))
                val imageTokens = imageTokensOut[0].toTensor().getDataAsFloatArray()
                saveFloatArrayToFile(File(executorchTempDir, "image_tokens.tensor"), imageTokens, longArrayOf(1, 577, 1024))
                module3.destroy()
            }
            System.gc()
            System.runFinalization()
            val part3Time = System.currentTimeMillis() - part3Start
            LogUtil.d(TAG, "Part 3 done in ${part3Time}ms, unloaded. Memory: ${getMemoryInfo()}")

            // Part 4: single .pte or chunked (4a_512 + 4a_65 + 4b) for lower peak RAM
            progressCallback?.invoke(0.60f, "Part 4: Decoder...")
            val imageTokensFile = File(executorchTempDir, "image_tokens.tensor")
            val latent0File = File(executorchTempDir, "latent0.tensor")
            val latent1File = File(executorchTempDir, "latent1.tensor")
            val x0FeatFile = File(executorchTempDir, "x0Feat.tensor")
            val x1FeatFile = File(executorchTempDir, "x1Feat.tensor")
            val x2File = File(executorchTempDir, "x2.tensor")
            val shapeLatent = longArrayOf(1, FEATURE_DIM.toLong(), mergedSize1x.toLong(), mergedSize1x.toLong())
            val shapeX1 = longArrayOf(1, FEATURE_DIM.toLong(), mergedSize05x.toLong(), mergedSize05x.toLong())
            val shapeX2 = longArrayOf(1, FEATURE_DIM.toLong(), SPATIAL_SIZE.toLong(), SPATIAL_SIZE.toLong())

            fun loadPart4Input(file: File, shape: LongArray): EValue {
                val (data, _) = loadFloatArrayFromFile(file)
                return EValue.from(Tensor.fromBlob(data, shape))
            }

            val part4Start = System.currentTimeMillis()
            val part4Output: Array<EValue>
            var part4ModuleToDestroy: Module? = null
            val useChunkedPart4 = isChunkedPart4Available()
            LogUtil.d(TAG, "Part 4: chunked available=$useChunkedPart4 (need part4a_chunk_512, part4a_chunk_65, part4b .pte)")

            if (useChunkedPart4) {
                // Chunked Part 4: ViT on token slices (512 + 65), then decoder once. Lowers peak RAM.
                val part4a512File = findFile(SPLIT_PART4A_CHUNK_512)!!
                val part4a65File = findFile(SPLIT_PART4A_CHUNK_65)!!
                val part4bFile = findFile(SPLIT_PART4B)!!
                LogUtil.d(TAG, "Part 4: chunked path (4a_512 + 4a_65 + 4b). Memory: ${getMemoryInfo()}")
                val (imageTokensFull, _) = loadFloatArrayFromFile(imageTokensFile)
                val chunk512Input = FloatArray(CHUNK_LEN_FIRST * FEATURE_DIM)
                System.arraycopy(imageTokensFull, 0, chunk512Input, 0, chunk512Input.size)
                System.gc()
                var module4a = Module.load(part4a512File.absolutePath, Module.LOAD_MODE_MMAP)
                val chunk512Output = module4a.forward(EValue.from(Tensor.fromBlob(chunk512Input, longArrayOf(1, CHUNK_LEN_FIRST.toLong(), FEATURE_DIM.toLong()))))
                val out512 = chunk512Output[0].toTensor().getDataAsFloatArray()
                module4a.destroy()
                module4a = null
                System.gc()
                System.runFinalization()
                val chunk65Input = FloatArray(CHUNK_LEN_LAST * FEATURE_DIM)
                System.arraycopy(imageTokensFull, CHUNK_LEN_FIRST * FEATURE_DIM, chunk65Input, 0, chunk65Input.size)
                System.gc()
                module4a = Module.load(part4a65File.absolutePath, Module.LOAD_MODE_MMAP)
                val chunk65Output = module4a.forward(EValue.from(Tensor.fromBlob(chunk65Input, longArrayOf(1, CHUNK_LEN_LAST.toLong(), FEATURE_DIM.toLong()))))
                val out65 = chunk65Output[0].toTensor().getDataAsFloatArray()
                module4a.destroy()
                module4a = null
                System.gc()
                System.runFinalization()
                val tokensAfterBlocks = FloatArray(IMAGE_TOKENS_SEQ_LEN * FEATURE_DIM)
                System.arraycopy(out512, 0, tokensAfterBlocks, 0, out512.size)
                System.arraycopy(out65, 0, tokensAfterBlocks, out512.size, out65.size)
                System.gc()
                // Part 4b signature: (tokens_after_blocks, image, latent0, latent1, x0_feat, x1_feat, x2_feat)
                LogUtil.d(TAG, "Part 4: running 4b (decoder). Memory: ${getMemoryInfo()}")
                val module4b = Module.load(part4bFile.absolutePath, Module.LOAD_MODE_MMAP)
                val ev1Tokens = EValue.from(Tensor.fromBlob(tokensAfterBlocks, longArrayOf(1, IMAGE_TOKENS_SEQ_LEN.toLong(), FEATURE_DIM.toLong())))
                System.gc()
                val ev2Image = loadPart4Input(inputImageFile, longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong()))
                System.gc()
                val ev3 = loadPart4Input(latent0File, shapeLatent)
                System.gc()
                val ev4 = loadPart4Input(latent1File, shapeLatent)
                System.gc()
                val ev5 = loadPart4Input(x0FeatFile, shapeLatent)
                System.gc()
                val ev6 = loadPart4Input(x1FeatFile, shapeX1)
                System.gc()
                val ev7 = loadPart4Input(x2File, shapeX2)
                System.gc()
                System.runFinalization()
                part4Output = try {
                    val out = module4b.forward(ev1Tokens, ev2Image, ev3, ev4, ev5, ev6, ev7)
                    LogUtil.d(TAG, "Part 4: 4b forward returned in ${System.currentTimeMillis() - part4Start}ms")
                    out
                } catch (e: OutOfMemoryError) {
                    LogUtil.e(TAG, "Part 4: OOM during 4b decoder forward.", e)
                    module4b.destroy()
                    System.gc()
                    progressCallback?.invoke(0f, "Out of memory during 3D decoder. Close other apps or use a device with 6GB+ RAM.")
                    return null
                }
                // Defer destroy until after PLY write: output buffer is zero-copy from module native memory
                part4ModuleToDestroy = module4b
            } else {
                // Single Part 4: load inputs one-by-one, one forward (can OOM on low-RAM devices)
                val part4File = findFile(SPLIT_PART4) ?: run {
                    LogUtil.e(TAG, "Missing ${SPLIT_PART4}")
                    return null
                }
                LogUtil.d(TAG, "Part 4: loading inputs one-by-one from disk. Memory: ${getMemoryInfo()}")
                System.gc()
                System.runFinalization()
                val module4 = Module.load(part4File.absolutePath, Module.LOAD_MODE_MMAP)
                part4ModuleToDestroy = module4
                System.gc()
                System.runFinalization()
                val ev1 = loadPart4Input(inputImageFile, longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong()))
                System.gc()
                val ev2 = loadPart4Input(imageTokensFile, longArrayOf(1, IMAGE_TOKENS_SEQ_LEN.toLong(), FEATURE_DIM.toLong()))
                System.gc()
                val ev3 = loadPart4Input(latent0File, shapeLatent)
                System.gc()
                val ev4 = loadPart4Input(latent1File, shapeLatent)
                System.gc()
                val ev5 = loadPart4Input(x0FeatFile, shapeLatent)
                System.gc()
                val ev6 = loadPart4Input(x1FeatFile, shapeX1)
                System.gc()
                val ev7 = loadPart4Input(x2File, shapeX2)
                System.gc()
                System.runFinalization()
                LogUtil.d(TAG, "Part 4: forward starting. Memory: ${getMemoryInfo()}")
                part4Output = try {
                    val out = module4.forward(ev1, ev2, ev3, ev4, ev5, ev6, ev7)
                    LogUtil.d(TAG, "Part 4: forward returned in ${System.currentTimeMillis() - part4Start}ms")
                    out
                } catch (e: OutOfMemoryError) {
                    LogUtil.e(TAG, "Part 4: OOM during decoder forward.", e)
                    module4.destroy()
                    part4ModuleToDestroy = null
                    System.gc()
                    progressCallback?.invoke(0f, "Out of memory during 3D decoder. Close other apps or use a device with 6GB+ RAM.")
                    return null
                }
            }

            val part4Time = System.currentTimeMillis() - part4Start
            LogUtil.d(TAG, "Part 4 done in ${part4Time}ms Memory: ${getMemoryInfo()}")

            val outputTensor = part4Output[0].toTensor()
            val outputBuffer = getTensorDataAsFloatBuffer(outputTensor)
            val gaussianCount = (outputTensor.numel() / PARAMS_PER_GAUSSIAN).toInt().coerceAtLeast(0)
            progressCallback?.invoke(0.80f, "Writing PLY ($gaussianCount Gaussians)...")
            val result = writeStreamingPlyPackedFromBuffer(gaussianCount, outputBuffer, progressCallback)
            outputBuffer.clear()
            part4ModuleToDestroy?.destroy()
            executorchTempDir.listFiles()?.forEach { it.delete() }
            System.gc()
            System.runFinalization()

            val elapsed = System.currentTimeMillis() - startTime
            LogUtil.d(TAG, "inferSplitMode SUCCESS totalMs=$elapsed gaussianCount=$gaussianCount (ONNX-style)")
            LogUtil.d(TAG, "  Breakdown: P1=${part12Saved.part1Time}ms P2=${part12Saved.part2Time}ms P3=${part3Time}ms P4=${part4Time}ms")

            progressCallback?.invoke(1.0f, "Done!")
            return result

        } catch (e: Exception) {
            LogUtil.e(TAG, "ExecuTorch SHARP split inference failed after ${System.currentTimeMillis() - startTime}ms", e)
            progressCallback?.invoke(0f, "Error: ${e.message}")
            return null
        }
    }


    /**
     * Reshape tokens [577, 1024] -> spatial [1024, 24, 24] by removing CLS token.
     */
    private fun reshapeTokensToSpatial(tokens: FloatArray): FloatArray {
        val spatial = FloatArray(FEATURE_DIM * SPATIAL_SIZE * SPATIAL_SIZE)
        for (h in 0 until SPATIAL_SIZE) {
            for (w in 0 until SPATIAL_SIZE) {
                val seqIdx = 1 + h * SPATIAL_SIZE + w  // skip CLS
                for (c in 0 until FEATURE_DIM) {
                    spatial[c * SPATIAL_SIZE * SPATIAL_SIZE + h * SPATIAL_SIZE + w] = tokens[seqIdx * FEATURE_DIM + c]
                }
            }
        }
        return spatial
    }

    /**
     * Full model inference: single forward pass through entire SHARP pipeline.
     * Input: [1, 3, 1536, 1536] -> Output: [N, 14] Gaussian parameters
     *
     * This is the CORRECT path - uses the complete encoder+decoder+head.
     */
    private suspend fun inferFullModel(
        bitmap: Bitmap,
        progressCallback: ((Float, String) -> Unit)?
    ): StreamingResult? {
        val startTime = System.currentTimeMillis()
        LogUtil.d(TAG, "inferFullModel ENTER bitmap=${bitmap.width}x${bitmap.height} Memory: ${getMemoryInfo()}")

        // Preprocess
        // #region agent log
        LogUtil.d("Progress0", """{"location":"ExecutorchSharp.kt:inferFullModel","message":"progress 0.05","data":{"progress":0.05,"msg":"Preprocessing"},"timestamp":${System.currentTimeMillis()}}""")
        // #endregion
        progressCallback?.invoke(0.05f, "Preprocessing image...")
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        val inputData = preprocessPatch(scaledBitmap) // reuse existing CHW preprocessing
        scaledBitmap.recycle()
        LogUtil.d(TAG, "Image preprocessed to ${IMAGE_SIZE}x${IMAGE_SIZE}")

        // Load full model
        // #region agent log
        LogUtil.d("Progress0", """{"location":"ExecutorchSharp.kt:inferFullModel","message":"progress 0.10","data":{"progress":0.1,"msg":"Loading model"},"timestamp":${System.currentTimeMillis()}}""")
        // #endregion
        progressCallback?.invoke(0.10f, "Loading full SHARP model...")
        val modelFile = findFullModel()
        if (modelFile == null) {
            LogUtil.e(TAG, "Full model not found (findFullModel returned null)")
            return null
        }
        // Encourage reclaim after a previous run to reduce OOM on second load
        System.gc()
        val loadStart = System.currentTimeMillis()
        val module = try {
            Module.load(modelFile.absolutePath, Module.LOAD_MODE_MMAP)
        } catch (e: OutOfMemoryError) {
            LogUtil.e(TAG, "Full model load OOM: ${e.message}")
            return null
        } catch (e: Throwable) {
            LogUtil.e(TAG, "Full model load failed: ${e.message}", e)
            return null
        }
        val loadTime = System.currentTimeMillis() - loadStart
        LogUtil.d(TAG, "Full model loaded (mmap): ${modelFile.name} in ${loadTime}ms")

        // Create input tensor - detect FP16 model and provide Half tensor
        val inputShape = longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong())
        val isFp16 = modelFile.name.contains("fp16") || modelFile.name.contains("memory_optimized")
        val inputTensor = if (isFp16) {
            val totalElements = 3 * IMAGE_SIZE * IMAGE_SIZE
            val halfBuffer = Tensor.allocateHalfBuffer(totalElements)
            for (v in inputData) {
                halfBuffer.put(halfFloatToShort(v))
            }
            halfBuffer.rewind()
            Tensor.fromBlob(halfBuffer, inputShape)
        } else {
            Tensor.fromBlob(inputData, inputShape)
        }
        LogUtil.d(TAG, "Input tensor created (fp16=$isFp16)")

        // Run inference (single forward pass - handles everything internally)
        // #region agent log
        LogUtil.d("Progress0", """{"location":"ExecutorchSharp.kt:inferFullModel","message":"progress 0.20, BEFORE forward","data":{"progress":0.2,"msg":"Running SHARP inference"},"timestamp":${System.currentTimeMillis()}}""")
        // #endregion
        progressCallback?.invoke(0.20f, "Running SHARP inference...")
        val inferStart = System.currentTimeMillis()
        val outputs = try {
            module.forward(EValue.from(inputTensor))
        } catch (e: OutOfMemoryError) {
            LogUtil.e(TAG, "Full model forward OOM: ${e.message}")
            module.destroy()
            return null
        } catch (e: Throwable) {
            LogUtil.e(TAG, "Full model forward failed: ${e.message}", e)
            module.destroy()
            return null
        }
        val inferTime = System.currentTimeMillis() - inferStart
        // #region agent log
        LogUtil.d("Progress0", """{"location":"ExecutorchSharp.kt:inferFullModel","message":"AFTER forward (inference done)","data":{"inferTimeMs":$inferTime},"timestamp":${System.currentTimeMillis()}}""")
        // #endregion
        LogUtil.d(TAG, "Full model inference completed in ${inferTime}ms")

        // Zero-copy: read from native FloatBuffer instead of getDataAsFloatArray (saves ~60MB JVM alloc)
        val outputTensor = outputs[0].toTensor()
        val outputBuffer = getTensorDataAsFloatBuffer(outputTensor)
        val gaussianCount = (outputTensor.numel() / PARAMS_PER_GAUSSIAN).toInt().coerceAtLeast(0)
        LogUtil.d(TAG, "Full model produced $gaussianCount Gaussians")

        if (gaussianCount <= 0) {
            LogUtil.e(TAG, "Full model produced 0 Gaussians")
            module.destroy()
            return null
        }

        // Convert to packed format for PLY writing (consume buffer before destroy)
        // #region agent log
        LogUtil.d("Progress0", """{"location":"ExecutorchSharp.kt:inferFullModel","message":"progress 0.75","data":{"progress":0.75,"gaussianCount":$gaussianCount},"timestamp":${System.currentTimeMillis()}}""")
        // #endregion
        progressCallback?.invoke(0.75f, "Writing PLY ($gaussianCount Gaussians)...")
        val result = writeStreamingPlyPackedFromBuffer(gaussianCount, outputBuffer, progressCallback)
        outputBuffer.clear()  // allow tensor to be reclaimed
        module.destroy()
        System.gc()

        val elapsed = System.currentTimeMillis() - startTime
        LogUtil.d(TAG, "inferFullModel SUCCESS totalMs=$elapsed gaussianCount=$gaussianCount")
        LogUtil.d(TAG, "  Breakdown: load=${loadTime}ms, infer=${inferTime}ms")

        // #region agent log
        LogUtil.d("Progress0", """{"location":"ExecutorchSharp.kt:inferFullModel","message":"progress 1.0 Done","data":{"progress":1.0},"timestamp":${System.currentTimeMillis()}}""")
        // #endregion
        progressCallback?.invoke(1.0f, "Done!")
        return result
    }

    /**
     * Zero-copy: get native FloatBuffer from Tensor via reflection (getRawDataBuffer).
     * Avoids getDataAsFloatArray which allocates ~60MB for 1.1M Gaussians.
     */
    private fun getTensorDataAsFloatBuffer(tensor: Tensor): FloatBuffer {
        return try {
            val method = tensor.javaClass.getDeclaredMethod("getRawDataBuffer")
            method.isAccessible = true
            when (val buf = method.invoke(tensor)) {
                is FloatBuffer -> buf
                is ByteBuffer -> buf.asFloatBuffer()
                else -> FloatBuffer.wrap(tensor.getDataAsFloatArray())
            }
        } catch (e: Exception) {
            LogUtil.w(TAG, "getRawDataBuffer failed, fallback to getDataAsFloatArray: ${e.message}")
            FloatBuffer.wrap(tensor.getDataAsFloatArray())
        }
    }

    /**
     * Encode a single 384x384 patch through the patch encoder.
     * Input: [1, 3, 384, 384] → Output: [1, 1024, 24, 24]
     * Returns flattened [1024, 24, 24] features as FloatArray.
     */
    private fun encodePatch(module: Module, patchBitmap: Bitmap): FloatArray {
        val inputData = preprocessPatch(patchBitmap)
        val inputTensor = Tensor.fromBlob(
            inputData,
            longArrayOf(1, 3, PATCH_SIZE.toLong(), PATCH_SIZE.toLong())
        )
        val outputs = module.forward(EValue.from(inputTensor))
        return outputs[0].toTensor().getDataAsFloatArray()
    }

    /**
     * Preprocess a 384x384 patch to CHW FloatArray normalized [0, 1].
     */
    private fun preprocessPatch(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val floatArray = FloatArray(3 * width * height)
        val channelSize = width * height

        for (i in pixels.indices) {
            floatArray[i] = ((pixels[i] shr 16) and 0xFF) / 255f
            floatArray[channelSize + i] = ((pixels[i] shr 8) and 0xFF) / 255f
            floatArray[2 * channelSize + i] = (pixels[i] and 0xFF) / 255f
        }
        return floatArray
    }

    /**
     * Compute merged grid output size.
     * First patch contributes full spatial size, subsequent patches contribute (spatial - 2*padding).
     */
    private fun getMergedSize(gridSize: Int, patchSpatial: Int, padding: Int): Int {
        val patchContrib = patchSpatial - 2 * padding
        return patchSpatial + (gridSize - 1) * patchContrib
    }

    /**
     * Merge patch features into a single spatial feature map.
     * Each patch is [1024, 24, 24] stored as a flat FloatArray in CHW order.
     * Output: [1024, outH, outW] stored as a flat FloatArray.
     *
     * Matching NCNN mergePatchGrid: uses overlap padding to blend edges.
     */
    private fun mergePatchGrid(
        patches: List<FloatArray>,
        gridSize: Int,
        padding: Int
    ): FloatArray {
        val patchH = SPATIAL_SIZE
        val patchW = SPATIAL_SIZE
        val outSize = getMergedSize(gridSize, patchH, padding)

        // Output: [FEATURE_DIM, outSize, outSize] in CHW format
        val output = FloatArray(FEATURE_DIM * outSize * outSize)

        var idx = 0
        var outY = 0

        for (gridJ in 0 until gridSize) {
            val srcY0 = if (gridJ == 0) 0 else padding
            val srcY1 = if (gridJ == gridSize - 1) patchH else (patchH - padding)
            val copyH = srcY1 - srcY0

            var outX = 0
            for (gridI in 0 until gridSize) {
                val patch = patches[idx++]
                val srcX0 = if (gridI == 0) 0 else padding
                val srcX1 = if (gridI == gridSize - 1) patchW else (patchW - padding)
                val copyW = srcX1 - srcX0

                // Copy region from patch to output for each channel
                for (c in 0 until FEATURE_DIM) {
                    val srcBase = c * patchH * patchW
                    val dstBase = c * outSize * outSize

                    for (dy in 0 until copyH) {
                        for (dx in 0 until copyW) {
                            val srcIdx = srcBase + (srcY0 + dy) * patchW + (srcX0 + dx)
                            val dstIdx = dstBase + (outY + dy) * outSize + (outX + dx)
                            output[dstIdx] = patch[srcIdx]
                        }
                    }
                }
                outX += copyW
            }
            outY += copyH
        }

        LogUtil.d(TAG, "Merged ${patches.size} patches into [${FEATURE_DIM}, $outSize, $outSize]")
        return output
    }

    /**
     * Extract Gaussians from GaussianHead output [1, 14, H, W].
     *
     * Channel layout:
     *   [0-2]: xyz position (raw values)
     *   [3]: opacity (apply sigmoid)
     *   [4-6]: scale (clamp to min)
     *   [7-10]: rotation quaternion (normalize)
     *   [11-13]: RGB color (clamp to [0,1])
     *
     * Output: packed [N, 14] in format: pos(3), scale(3), rot(4), opacity(1), color(3)
     */
    private fun extractGaussians(headOutput: FloatArray, outH: Int, outW: Int): FloatArray {
        val numGaussians = outH * outW
        val channelStride = outH * outW
        val params = FloatArray(numGaussians * PARAMS_PER_GAUSSIAN)

        for (pixIdx in 0 until numGaussians) {
            val offset = pixIdx * PARAMS_PER_GAUSSIAN

            // Position (channels 0-2, raw)
            params[offset + 0] = headOutput[0 * channelStride + pixIdx]
            params[offset + 1] = headOutput[1 * channelStride + pixIdx]
            params[offset + 2] = headOutput[2 * channelStride + pixIdx]

            // Scale (channels 4-6, clamp to min 0.001)
            params[offset + 3] = max(0.001f, headOutput[4 * channelStride + pixIdx])
            params[offset + 4] = max(0.001f, headOutput[5 * channelStride + pixIdx])
            params[offset + 5] = max(0.001f, headOutput[6 * channelStride + pixIdx])

            // Rotation (channels 7-10, normalize quaternion)
            val qw = headOutput[7 * channelStride + pixIdx]
            val qx = headOutput[8 * channelStride + pixIdx]
            val qy = headOutput[9 * channelStride + pixIdx]
            val qz = headOutput[10 * channelStride + pixIdx]
            val qnorm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            val invNorm = if (qnorm > 1e-6f) 1f / qnorm else 1f
            params[offset + 6] = qw * invNorm
            params[offset + 7] = qx * invNorm
            params[offset + 8] = qy * invNorm
            params[offset + 9] = qz * invNorm

            // Opacity (channel 3, apply sigmoid)
            val rawOpacity = headOutput[3 * channelStride + pixIdx]
            params[offset + 10] = 1f / (1f + kotlin.math.exp(-rawOpacity))

            // Color (channels 11-13, model outputs BGR; swap to RGB)
            params[offset + 11] = headOutput[13 * channelStride + pixIdx].coerceIn(0f, 1f)
            params[offset + 12] = headOutput[12 * channelStride + pixIdx].coerceIn(0f, 1f)
            params[offset + 13] = headOutput[11 * channelStride + pixIdx].coerceIn(0f, 1f)
        }

        return params
    }

    /**
     * Zero-copy variant: read from FloatBuffer (native-backed) instead of FloatArray.
     * Avoids ~60MB JVM allocation for full model output.
     */
    private fun writeStreamingPlyPackedFromBuffer(
        gaussianCount: Int,
        params: FloatBuffer,
        progressCallback: ((Float, String) -> Unit)?
    ): StreamingResult {
        LogUtil.d(TAG, "writeStreamingPlyPackedFromBuffer ENTER gaussianCount=$gaussianCount (zero-copy)")
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

            val batchBuffer = plyBatch; batchBuffer.clear()
            val scaleBoost = 1.3f
            val minScale = 0.001f
            val lutScale = (LOGIT_LUT_SIZE - 1).toFloat()

            val progressEvery = max(1, gaussianCount / 10)
            var processed = 0

            while (processed < gaussianCount) {
                val currentBatch = minOf(PLY_BATCH_SIZE, gaussianCount - processed)

                for (j in 0 until currentBatch) {
                    val offset = (processed + j) * PARAMS_PER_GAUSSIAN

                    val x = params.get(offset + 0)
                    val y = -params.get(offset + 1)
                    val z = -params.get(offset + 2)

                    if (x < minX) minX = x; if (x > maxX) maxX = x
                    if (y < minY) minY = y; if (y > maxY) maxY = y
                    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z

                    batchBuffer.putFloat(x)
                    batchBuffer.putFloat(y)
                    batchBuffer.putFloat(z)

                    batchBuffer.putFloat(0f); batchBuffer.putFloat(0f); batchBuffer.putFloat(0f)

                    // Packed layout from Python: [positions(0-2), opacity(3), scales(4-6), quaternions(7-10), colors(11-13)]
                    val r = params.get(offset + 11).coerceIn(0f, 1f)
                    val g = params.get(offset + 12).coerceIn(0f, 1f)
                    val b = params.get(offset + 13).coerceIn(0f, 1f)
                    batchBuffer.putFloat((r - 0.5f) / SH_C0)
                    batchBuffer.putFloat((g - 0.5f) / SH_C0)
                    batchBuffer.putFloat((b - 0.5f) / SH_C0)

                    zeroSHBuffer.clear(); batchBuffer.put(zeroSHBuffer)

                    val rawOpacity = params.get(offset + 3).coerceIn(0f, 1f)
                    val lutIndex = (rawOpacity * lutScale).toInt().coerceIn(0, LOGIT_LUT_SIZE - 1)
                    batchBuffer.putFloat(LOGIT_LUT[lutIndex])

                    batchBuffer.putFloat(lnLut(max(params.get(offset + 4) * scaleBoost, minScale)))
                    batchBuffer.putFloat(lnLut(max(params.get(offset + 5) * scaleBoost, minScale)))
                    batchBuffer.putFloat(lnLut(max(params.get(offset + 6) * scaleBoost, minScale)))

                    val rw = params.get(offset + 7)
                    val rx = params.get(offset + 8)
                    val ry = params.get(offset + 9)
                    val rz = params.get(offset + 10)
                    val mag = sqrt(rw * rw + rx * rx + ry * ry + rz * rz)
                    val invMag = if (mag > 1e-8f) 1f / mag else 1f
                    batchBuffer.putFloat(rw * invMag)
                    batchBuffer.putFloat(rx * invMag)
                    batchBuffer.putFloat(ry * invMag)
                    batchBuffer.putFloat(rz * invMag)
                }

                batchBuffer.flip()
                batchBuffer.limit(currentBatch * BYTES_PER_VERTEX)
                while (batchBuffer.hasRemaining()) {
                    channel.write(batchBuffer)
                }
                batchBuffer.clear()

                processed += currentBatch
                if (processed % progressEvery == 0 || processed == gaussianCount) {
                    val progress = 0.75f + (processed.toFloat() / gaussianCount) * 0.20f
                    progressCallback?.invoke(progress, "Writing PLY ($processed/$gaussianCount)...")
                }
            }
        }

        plyFile.copyTo(classicPlyFile, overwrite = true)

        LogUtil.d(TAG, "writeStreamingPlyPackedFromBuffer DONE file=${plyFile.absolutePath} bytes=${plyFile.length()}")
        LogUtil.d(TAG, "Room bounds: ${maxX - minX}m x ${maxY - minY}m x ${maxZ - minZ}m")

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
     * Write packed [N, 14] Gaussian output to PLY file.
     * Layout per gaussian: pos(3), scale(3), rot(4), opacity(1), color(3)
     */
    private fun writeStreamingPlyPacked(
        gaussianCount: Int,
        params: FloatArray,
        progressCallback: ((Float, String) -> Unit)?
    ): StreamingResult {
        LogUtil.d(TAG, "writeStreamingPlyPacked ENTER gaussianCount=$gaussianCount paramsLen=${params.size}")
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

            val batchBuffer = plyBatch; batchBuffer.clear()
            val scaleBoost = 1.3f
            val minScale = 0.001f
            val lutScale = (LOGIT_LUT_SIZE - 1).toFloat()

            val progressEvery = max(1, gaussianCount / 10)
            var processed = 0

            while (processed < gaussianCount) {
                val currentBatch = minOf(PLY_BATCH_SIZE, gaussianCount - processed)

                for (j in 0 until currentBatch) {
                    val offset = (processed + j) * PARAMS_PER_GAUSSIAN

                    val x = params[offset + 0]
                    val y = -params[offset + 1]
                    val z = -params[offset + 2]

                    if (x < minX) minX = x; if (x > maxX) maxX = x
                    if (y < minY) minY = y; if (y > maxY) maxY = y
                    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z

                    batchBuffer.putFloat(x)
                    batchBuffer.putFloat(y)
                    batchBuffer.putFloat(z)

                    // Normals
                    batchBuffer.putFloat(0f)
                    batchBuffer.putFloat(0f)
                    batchBuffer.putFloat(0f)

                    // Colors (at offset 11-13) -> SH DC
                    val r = params[offset + 11].coerceIn(0f, 1f)
                    val g = params[offset + 12].coerceIn(0f, 1f)
                    val b = params[offset + 13].coerceIn(0f, 1f)
                    batchBuffer.putFloat((r - 0.5f) / SH_C0)
                    batchBuffer.putFloat((g - 0.5f) / SH_C0)
                    batchBuffer.putFloat((b - 0.5f) / SH_C0)

                    // Higher order SH (45 zeros)
                    zeroSHBuffer.clear(); batchBuffer.put(zeroSHBuffer)

                    // Opacity (at offset 10) -> logit via LUT
                    val rawOpacity = params[offset + 10].coerceIn(0f, 1f)
                    val lutIndex = (rawOpacity * lutScale).toInt().coerceIn(0, LOGIT_LUT_SIZE - 1)
                    batchBuffer.putFloat(LOGIT_LUT[lutIndex])

                    // Scale (at offset 3-5) -> log via LN LUT
                    batchBuffer.putFloat(lnLut(max(params[offset + 3] * scaleBoost, minScale)))
                    batchBuffer.putFloat(lnLut(max(params[offset + 4] * scaleBoost, minScale)))
                    batchBuffer.putFloat(lnLut(max(params[offset + 5] * scaleBoost, minScale)))

                    // Rotation (at offset 6-9) -> normalize
                    val rw = params[offset + 6]
                    val rx = params[offset + 7]
                    val ry = params[offset + 8]
                    val rz = params[offset + 9]
                    val mag = sqrt(rw * rw + rx * rx + ry * ry + rz * rz)
                    val invMag = if (mag > 1e-8f) 1f / mag else 1f
                    batchBuffer.putFloat(rw * invMag)
                    batchBuffer.putFloat(rx * invMag)
                    batchBuffer.putFloat(ry * invMag)
                    batchBuffer.putFloat(rz * invMag)
                }

                batchBuffer.flip()
                batchBuffer.limit(currentBatch * BYTES_PER_VERTEX)
                while (batchBuffer.hasRemaining()) {
                    channel.write(batchBuffer)
                }
                batchBuffer.clear()

                processed += currentBatch
                if (processed % progressEvery == 0 || processed == gaussianCount) {
                    val progress = 0.75f + (processed.toFloat() / gaussianCount) * 0.20f
                    progressCallback?.invoke(progress, "Writing PLY ($processed/$gaussianCount)...")
                }
            }
        }

        plyFile.copyTo(classicPlyFile, overwrite = true)

        LogUtil.d(TAG, "writeStreamingPlyPacked DONE file=${plyFile.absolutePath} bytes=${plyFile.length()}")
        LogUtil.d(TAG, "Room bounds: ${maxX - minX}m x ${maxY - minY}m x ${maxZ - minZ}m min=($minX,$minY,$minZ) max=($maxX,$maxY,$maxZ)")

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

    /**
     * Convert a 32-bit float to a 16-bit half-precision float stored as Short.
     * IEEE 754 half-precision: 1 sign + 5 exponent + 10 mantissa bits.
     */
    private fun halfFloatToShort(floatValue: Float): Short {
        val intBits = java.lang.Float.floatToIntBits(floatValue)
        val sign = (intBits ushr 16) and 0x8000
        val exponent = ((intBits ushr 23) and 0xFF) - 127 + 15
        val mantissa = intBits and 0x7FFFFF

        val halfBits = when {
            exponent <= 0 -> sign // too small, flush to zero
            exponent >= 31 -> sign or 0x7C00 // overflow to infinity
            else -> sign or (exponent shl 10) or (mantissa ushr 13)
        }
        return halfBits.toShort()
    }

    fun release() {
        isInitialized = false
        LogUtil.d(TAG, "ExecutorchSharp released")
    }
}
