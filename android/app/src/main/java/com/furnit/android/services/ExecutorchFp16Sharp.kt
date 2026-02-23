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
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/**
 * ExecuTorch FP16 SHARP inference via 4-part split .pte models with XNNPACK.
 *
 * Same pipeline as ExecutorchSharp split mode but with FP16 models (~50% smaller).
 * Model I/O uses FP16 tensors; intermediate merging stays FP32 for precision.
 *
 * Models: sharp_split_part{1,2,3,4}_fp16.pte
 */
class ExecutorchFp16Sharp private constructor(private val context: Context) {

    companion object {
        private const val TAG = "ExecutorchFp16Sharp"
        private const val IMAGE_SIZE = 1536
        private const val PATCH_SIZE = 384
        private const val FEATURE_DIM = 1024
        private const val SPATIAL_SIZE = 24
        private const val PARAMS_PER_GAUSSIAN = 14
        private const val SH_C0 = 0.28209479177387814f
        private const val BYTES_PER_VERTEX = 62 * 4
        private const val PLY_BATCH_SIZE = 512

        private const val GRID_1X = 5
        private const val GRID_05X = 3
        private const val TOTAL_PATCHES = 35
        private const val PADDING_1X = 3
        private const val PADDING_05X = 6

        private const val SPLIT_PART1 = "sharp_split_part1_fp16.pte"
        private const val SPLIT_PART2 = "sharp_split_part2_fp16.pte"
        private const val SPLIT_PART3 = "sharp_split_part3_fp16.pte"
        private const val SPLIT_PART4 = "sharp_split_part4_fp16.pte"
        private val SPLIT_FILENAMES = arrayOf(SPLIT_PART1, SPLIT_PART2, SPLIT_PART3, SPLIT_PART4)

        private const val SPLIT_PART4A_CHUNK_512 = "sharp_split_part4a_chunk_512_fp16.pte"
        private const val SPLIT_PART4A_CHUNK_65 = "sharp_split_part4a_chunk_65_fp16.pte"
        private const val SPLIT_PART4B = "sharp_split_part4b_fp16.pte"
        private const val IMAGE_TOKENS_SEQ_LEN = 577
        private const val CHUNK_LEN_FIRST = 512
        private const val CHUNK_LEN_LAST = IMAGE_TOKENS_SEQ_LEN - CHUNK_LEN_FIRST

        private val EXTRA_SEARCH_DIRS = arrayOf("/data/local/tmp/furnit/")

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
        private var instance: ExecutorchFp16Sharp? = null
        private val mutex = Mutex()

        fun getInstance(context: Context): ExecutorchFp16Sharp {
            return instance ?: synchronized(this) {
                instance ?: ExecutorchFp16Sharp(context.applicationContext).also {
                    instance = it
                    Log.d(TAG, "ExecutorchFp16Sharp singleton created")
                }
            }
        }
    }

    @Volatile
    private var isInitialized = false

    private val zeroSHBuffer: ByteBuffer by lazy {
        ByteBuffer.allocateDirect(45 * 4).apply { order(ByteOrder.LITTLE_ENDIAN) }
    }
    private val plyBatch: ByteBuffer by lazy {
        ByteBuffer.allocateDirect(BYTES_PER_VERTEX * PLY_BATCH_SIZE).apply { order(ByteOrder.LITTLE_ENDIAN) }
    }

    private val internalModelsDir: File by lazy {
        File(context.filesDir, "models").also { it.mkdirs() }
    }
    private val externalModelsDir: File? get() = context.getExternalFilesDir("models")
    private val tempDir: File by lazy {
        File(context.cacheDir, "executorch_fp16_temp").also { it.mkdirs() }
    }

    data class StreamingResult(
        val plyFile: File,
        val classicPlyFile: File,
        val gaussianCount: Int,
        val roomWidth: Float,
        val roomHeight: Float,
        val roomDepth: Float
    )

    // ---- File lookup ----

    private fun findFile(filename: String): File? {
        val internal = File(internalModelsDir, filename)
        if (internal.exists() && internal.length() > 0) return internal
        externalModelsDir?.let { ext ->
            val external = File(ext, filename)
            if (external.exists() && external.length() > 0) return external
        }
        for (dir in EXTRA_SEARCH_DIRS) {
            val file = File(dir, filename)
            if (file.exists() && file.length() > 0) return file
        }
        return null
    }

    fun isModelReady(): Boolean {
        val ready = SPLIT_FILENAMES.all { findFile(it) != null }
        Log.d(TAG, "isModelReady: $ready")
        return ready
    }

    fun getMissingFiles(): List<String> = SPLIT_FILENAMES.filter { findFile(it) == null }

    fun getModelsDirPath(): String = externalModelsDir?.absolutePath ?: internalModelsDir.absolutePath

    private fun isChunkedPart4Available(): Boolean {
        return findFile(SPLIT_PART4A_CHUNK_512) != null &&
            findFile(SPLIT_PART4A_CHUNK_65) != null &&
            findFile(SPLIT_PART4B) != null
    }

    // ---- Internal storage copy (fast mmap) ----

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

    private fun ensureSplitModelsInInternalStorage(): Boolean {
        for (filename in SPLIT_FILENAMES) {
            val dst = File(internalModelsDir, filename)
            val src = findSourceForCopy(filename) ?: if (dst.exists() && dst.length() > 0) continue else {
                Log.w(TAG, "ensureSplitModels: no source for $filename")
                return false
            }
            if (dst.exists() && dst.length() == src.length()) continue
            try {
                Log.d(TAG, "Copying $filename (${src.length() / 1024 / 1024} MB) to internal...")
                val tmp = File(internalModelsDir, "$filename.tmp")
                FileInputStream(src).use { fis ->
                    FileOutputStream(tmp).use { fos ->
                        val buf = ByteArray(1024 * 1024)
                        while (true) {
                            val n = fis.read(buf)
                            if (n <= 0) break
                            fos.write(buf, 0, n)
                        }
                        fos.fd.sync()
                    }
                }
                if (tmp.length() != src.length()) { tmp.delete(); return false }
                if (!tmp.renameTo(dst)) { tmp.delete(); return false }
                Log.d(TAG, "Copied $filename to ${dst.absolutePath}")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to copy $filename: ${e.message}")
                File(internalModelsDir, "$filename.tmp").delete()
                return false
            }
        }
        return true
    }

    // ---- Initialize ----

    private fun initializeImpl(): Boolean {
        Log.d(TAG, "initialize ENTER. Memory: ${getMemoryInfo()}")
        if (!ensureSplitModelsInInternalStorage()) {
            Log.e(TAG, "initialize: copy to internal failed")
        }
        if (areAllSplitModelsInternal()) {
            for (fn in SPLIT_FILENAMES) {
                val f = File(internalModelsDir, fn)
                Log.d(TAG, "  FP16 split: ${f.name} (${f.length() / 1024 / 1024}MB)")
            }
            isInitialized = true
            return true
        }
        if (SPLIT_FILENAMES.all { findFile(it) != null }) {
            isInitialized = true
            return true
        }
        Log.e(TAG, "No FP16 ExecuTorch models found")
        return false
    }

    fun initialize(): Boolean = runBlocking { mutex.withLock { initializeImpl() } }

    // ---- Preload Part 1 ----

    suspend fun preloadAndWarmup(progress: ((String) -> Unit)? = null) = withContext(Dispatchers.IO) {
        mutex.withLock {
            if (!isInitialized) initializeImpl()
        }
        val part1File = findFile(SPLIT_PART1) ?: return@withContext
        Log.d(TAG, "preloadAndWarmup: validating Part1 FP16 (${part1File.length() / 1024 / 1024}MB)")
        progress?.invoke("Validating FP16 encoder A...")
        val t0 = System.currentTimeMillis()
        val module = Module.load(part1File.absolutePath, Module.LOAD_MODE_MMAP)
        Log.d(TAG, "PRELOAD FP16 Part1 load ${System.currentTimeMillis() - t0}ms")
        module.destroy()
        System.gc()
        Log.d(TAG, "preloadAndWarmup DONE")
        progress?.invoke("Preload done")
    }

    // ---- FP16 tensor helpers ----

    private fun halfFloatToShort(floatValue: Float): Short {
        val intBits = java.lang.Float.floatToIntBits(floatValue)
        val sign = (intBits ushr 16) and 0x8000
        val exponent = ((intBits ushr 23) and 0xFF) - 127 + 15
        val mantissa = intBits and 0x7FFFFF
        val halfBits = when {
            exponent <= 0 -> sign
            exponent >= 31 -> sign or 0x7C00
            else -> sign or (exponent shl 10) or (mantissa ushr 13)
        }
        return halfBits.toShort()
    }

    private fun halfShortToFloat(halfBits: Short): Float {
        val h = halfBits.toInt() and 0xFFFF
        val sign = (h ushr 15) and 1
        val exponent = (h ushr 10) and 0x1F
        val mantissa = h and 0x3FF
        val floatBits = when {
            exponent == 0 -> if (mantissa == 0) sign shl 31 else {
                var m = mantissa
                var e = -14
                while (m and 0x400 == 0) { m = m shl 1; e-- }
                m = m and 0x3FF
                (sign shl 31) or ((e + 127) shl 23) or (m shl 13)
            }
            exponent == 31 -> (sign shl 31) or (0xFF shl 23) or (mantissa shl 13)
            else -> (sign shl 31) or ((exponent - 15 + 127) shl 23) or (mantissa shl 13)
        }
        return java.lang.Float.intBitsToFloat(floatBits)
    }

    /**
     * Create an FP16 tensor from FP32 data.
     */
    private fun createHalfTensor(floatData: FloatArray, shape: LongArray): Tensor {
        val halfBuffer = Tensor.allocateHalfBuffer(floatData.size)
        for (v in floatData) halfBuffer.put(halfFloatToShort(v))
        halfBuffer.rewind()
        return Tensor.fromBlob(halfBuffer, shape)
    }

    /**
     * Extract float data from a tensor that may be FP16 or FP32.
     * Tries getDataAsFloatArray first; falls back to reading raw half buffer.
     */
    private fun getOutputAsFloatArray(tensor: Tensor): FloatArray {
        return try {
            tensor.getDataAsFloatArray()
        } catch (e: Exception) {
            Log.d(TAG, "getDataAsFloatArray failed (FP16 tensor?), converting manually: ${e.message}")
            try {
                val method = tensor.javaClass.getDeclaredMethod("getRawDataBuffer")
                method.isAccessible = true
                when (val buf = method.invoke(tensor)) {
                    is ShortBuffer -> {
                        buf.rewind()
                        FloatArray(buf.remaining()) { halfShortToFloat(buf.get()) }
                    }
                    is ByteBuffer -> {
                        buf.order(ByteOrder.LITTLE_ENDIAN)
                        val shortBuf = buf.asShortBuffer()
                        FloatArray(shortBuf.remaining()) { halfShortToFloat(shortBuf.get()) }
                    }
                    is FloatBuffer -> {
                        buf.rewind()
                        FloatArray(buf.remaining()) { buf.get() }
                    }
                    else -> throw RuntimeException("Unknown buffer type: ${buf?.javaClass}")
                }
            } catch (e2: Exception) {
                Log.e(TAG, "Manual FP16 read also failed: ${e2.message}")
                throw e2
            }
        }
    }

    /**
     * Get a FloatBuffer from a tensor (may be FP16 or FP32). For PLY writing.
     */
    private fun getTensorDataAsFloatBuffer(tensor: Tensor): FloatBuffer {
        return try {
            val method = tensor.javaClass.getDeclaredMethod("getRawDataBuffer")
            method.isAccessible = true
            when (val buf = method.invoke(tensor)) {
                is FloatBuffer -> buf
                is ByteBuffer -> {
                    buf.order(ByteOrder.LITTLE_ENDIAN)
                    if (buf.remaining() % 2 == 0 && buf.remaining() != buf.remaining() / 2 * 4) {
                        val shortBuf = buf.asShortBuffer()
                        val floatArr = FloatArray(shortBuf.remaining()) { halfShortToFloat(shortBuf.get()) }
                        FloatBuffer.wrap(floatArr)
                    } else {
                        buf.asFloatBuffer()
                    }
                }
                is ShortBuffer -> {
                    buf.rewind()
                    val floatArr = FloatArray(buf.remaining()) { halfShortToFloat(buf.get()) }
                    FloatBuffer.wrap(floatArr)
                }
                else -> FloatBuffer.wrap(tensor.getDataAsFloatArray())
            }
        } catch (e: Exception) {
            Log.w(TAG, "getTensorDataAsFloatBuffer fallback: ${e.message}")
            FloatBuffer.wrap(getOutputAsFloatArray(tensor))
        }
    }

    // ---- Intermediate I/O (same format as ExecutorchSharp) ----

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
    }

    private fun loadFloatArrayFromFile(file: File): Pair<FloatArray, LongArray> {
        RandomAccessFile(file, "r").use { raf ->
            val channel = raf.channel
            val header = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
            channel.read(header); header.flip()
            val numDims = header.int
            val shapeBuf = ByteBuffer.allocate(8 * numDims).order(ByteOrder.LITTLE_ENDIAN)
            channel.read(shapeBuf); shapeBuf.flip()
            val shape = LongArray(numDims) { shapeBuf.long }
            val totalFloats = shape.fold(1L) { acc, d -> acc * d }.toInt()
            val data = FloatArray(totalFloats)
            val dataBuf = channel.map(FileChannel.MapMode.READ_ONLY, 4L + 8 * numDims, totalFloats * 4L)
                .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            dataBuf.get(data)
            return data to shape
        }
    }

    // ---- Memory / diagnostics ----

    private fun getMemoryInfo(): String {
        return try {
            val runtime = Runtime.getRuntime()
            val usedMB = (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024
            val maxMB = runtime.maxMemory() / 1024 / 1024
            val availSysMB = getAvailSysMemMB()
            "JVM: ${usedMB}/${maxMB}MB, SysAvail: ${availSysMB}MB"
        } catch (_: Exception) {
            "JVM: ${(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024 / 1024}MB"
        }
    }

    private fun getAvailSysMemMB(): Long {
        return try {
            val am = context.getSystemService(android.content.Context.ACTIVITY_SERVICE) as? android.app.ActivityManager
            am?.let { actMgr ->
                val memInfo = android.app.ActivityManager.MemoryInfo()
                actMgr.getMemoryInfo(memInfo)
                memInfo.availMem / 1024 / 1024
            } ?: -1L
        } catch (_: Exception) { -1L }
    }

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

    private fun getMergedSize(gridSize: Int, patchSpatial: Int, padding: Int): Int {
        return patchSpatial + (gridSize - 1) * (patchSpatial - 2 * padding)
    }

    // ---- Inference ----

    suspend fun inferStreaming(
        bitmap: Bitmap,
        progressCallback: ((Float, String) -> Unit)? = null
    ): StreamingResult? = withContext(Dispatchers.IO) {
        mutex.withLock {
            if (!isInitialized) {
                Log.e(TAG, "Not initialized")
                return@withContext null
            }
            Log.d(TAG, "inferStreaming ENTER (FP16) bitmap=${bitmap.width}x${bitmap.height} Memory: ${getMemoryInfo()}")
            return@withContext inferSplitMode(bitmap, progressCallback)
        }
    }

    private suspend fun inferSplitMode(
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

        val tempSpatial = FloatArray(FEATURE_DIM * SPATIAL_SIZE * SPATIAL_SIZE)

        fun reshapeTokensToSpatialInto(tokens: FloatArray, spatialOut: FloatArray) {
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
            Log.d(TAG, "inferSplitMode ENTER (FP16) Memory: ${getMemoryInfo()}")

            progressCallback?.invoke(0.02f, "Preprocessing...")
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
            val halfSize = IMAGE_SIZE / 2
            val halfBitmap = Bitmap.createScaledBitmap(scaledBitmap, halfSize, halfSize, true)
            val quarterBitmap = Bitmap.createScaledBitmap(scaledBitmap, PATCH_SIZE, PATCH_SIZE, true)

            val mergedSize1x = getMergedSize(GRID_1X, SPATIAL_SIZE, PADDING_1X)
            val mergedSize05x = getMergedSize(GRID_05X, SPATIAL_SIZE, PADDING_05X)
            val stride1x = (IMAGE_SIZE - PATCH_SIZE) / 4
            val stride05x = (halfSize - PATCH_SIZE) / 2

            // Part 1+2 block: stream per-patch, merge on-the-fly, save to disk
            data class Part12Saved(val inputImageFile: File, val mergedSize1x: Int, val mergedSize05x: Int, val elapsedMs: Long)
            val part12Saved = run {
                val latent0 = FloatArray(FEATURE_DIM * mergedSize1x * mergedSize1x)
                val latent1 = FloatArray(FEATURE_DIM * mergedSize1x * mergedSize1x)
                val x0Feat = FloatArray(FEATURE_DIM * mergedSize1x * mergedSize1x)
                val x1Feat = FloatArray(FEATURE_DIM * mergedSize05x * mergedSize05x)
                var x2Feat: FloatArray? = null
                val colOffsets1x = buildColumnOffsets(GRID_1X, PADDING_1X)
                val colOffsets05x = buildColumnOffsets(GRID_05X, PADDING_05X)
                val rowOffset1xL0 = intArrayOf(0)
                val rowOffset1xL1 = intArrayOf(0)
                val rowOffset1xX0 = intArrayOf(0)
                val rowOffset05x = intArrayOf(0)

                progressCallback?.invoke(0.05f, "Loading FP16 encoder A + B...")
                val part1File = findFile(SPLIT_PART1) ?: return null
                val part2File = findFile(SPLIT_PART2) ?: return null
                val tLoad0 = System.currentTimeMillis()
                val module1 = Module.load(part1File.absolutePath, Module.LOAD_MODE_MMAP)
                val module2 = Module.load(part2File.absolutePath, Module.LOAD_MODE_MMAP)
                Log.d(TAG, "FP16 Part1+Part2 load done in ${System.currentTimeMillis() - tLoad0}ms")

                var patchCount = 0
                val part12Start = System.currentTimeMillis()

                // 1x patches (5x5)
                for (i in 0 until GRID_1X) {
                    for (j in 0 until GRID_1X) {
                        val patchBitmap = Bitmap.createBitmap(scaledBitmap, j * stride1x, i * stride1x, PATCH_SIZE, PATCH_SIZE)
                        val patchData = preprocessPatch(patchBitmap)
                        patchBitmap.recycle()

                        val inputTensor = createHalfTensor(patchData, longArrayOf(1, 3, PATCH_SIZE.toLong(), PATCH_SIZE.toLong()))
                        val out1 = module1.forward(EValue.from(inputTensor))
                        val tokens = getOutputAsFloatArray(out1[0].toTensor())
                        val block5 = getOutputAsFloatArray(out1[1].toTensor())

                        reshapeTokensToSpatialInto(block5, tempSpatial)
                        mergePatchCropInto(latent0, mergedSize1x, tempSpatial, j, i, GRID_1X, PADDING_1X, rowOffset1xL0, colOffsets1x)
                        reshapeTokensToSpatialInto(tokens, tempSpatial)
                        mergePatchCropInto(latent1, mergedSize1x, tempSpatial, j, i, GRID_1X, PADDING_1X, rowOffset1xL1, colOffsets1x)

                        val tokensTensor = createHalfTensor(tokens, longArrayOf(1, 577, 1024))
                        val feat = getOutputAsFloatArray(module2.forward(EValue.from(tokensTensor))[0].toTensor())
                        mergePatchCropInto(x0Feat, mergedSize1x, feat, j, i, GRID_1X, PADDING_1X, rowOffset1xX0, colOffsets1x)

                        patchCount++
                        if (patchCount % 5 == 0) {
                            progressCallback?.invoke(0.05f + (patchCount.toFloat() / TOTAL_PATCHES) * 0.40f, "FP16 Part 1+2: Patch $patchCount/$TOTAL_PATCHES...")
                        }
                    }
                }
                // 0.5x patches (3x3)
                for (i in 0 until GRID_05X) {
                    for (j in 0 until GRID_05X) {
                        val patchBitmap = Bitmap.createBitmap(halfBitmap, j * stride05x, i * stride05x, PATCH_SIZE, PATCH_SIZE)
                        val patchData = preprocessPatch(patchBitmap)
                        patchBitmap.recycle()

                        val inputTensor = createHalfTensor(patchData, longArrayOf(1, 3, PATCH_SIZE.toLong(), PATCH_SIZE.toLong()))
                        val tokens = getOutputAsFloatArray(module1.forward(EValue.from(inputTensor))[0].toTensor())
                        val tokensTensor = createHalfTensor(tokens, longArrayOf(1, 577, 1024))
                        val feat = getOutputAsFloatArray(module2.forward(EValue.from(tokensTensor))[0].toTensor())
                        mergePatchCropInto(x1Feat, mergedSize05x, feat, j, i, GRID_05X, PADDING_05X, rowOffset05x, colOffsets05x)
                        patchCount++
                    }
                }
                halfBitmap.recycle()

                // 0.25x patch
                val qData = preprocessPatch(quarterBitmap)
                quarterBitmap.recycle()
                val qInput = createHalfTensor(qData, longArrayOf(1, 3, PATCH_SIZE.toLong(), PATCH_SIZE.toLong()))
                val qTokens = getOutputAsFloatArray(module1.forward(EValue.from(qInput))[0].toTensor())
                val qTokensTensor = createHalfTensor(qTokens, longArrayOf(1, 577, 1024))
                x2Feat = getOutputAsFloatArray(module2.forward(EValue.from(qTokensTensor))[0].toTensor())
                patchCount++

                module1.destroy(); module2.destroy()
                System.gc(); System.runFinalization()
                Log.d(TAG, "FP16 Part 1+2 done: $patchCount patches in ${System.currentTimeMillis() - part12Start}ms Memory: ${getMemoryInfo()}")

                val x2 = x2Feat ?: return null
                tempDir.listFiles()?.forEach { it.delete() }
                saveFloatArrayToFile(File(tempDir, "latent0.tensor"), latent0, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize1x.toLong(), mergedSize1x.toLong()))
                saveFloatArrayToFile(File(tempDir, "latent1.tensor"), latent1, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize1x.toLong(), mergedSize1x.toLong()))
                saveFloatArrayToFile(File(tempDir, "x0Feat.tensor"), x0Feat, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize1x.toLong(), mergedSize1x.toLong()))
                saveFloatArrayToFile(File(tempDir, "x1Feat.tensor"), x1Feat, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize05x.toLong(), mergedSize05x.toLong()))
                saveFloatArrayToFile(File(tempDir, "x2.tensor"), x2, longArrayOf(1, FEATURE_DIM.toLong(), SPATIAL_SIZE.toLong(), SPATIAL_SIZE.toLong()))

                val imageData = preprocessPatch(scaledBitmap)
                val inputImageFile = File(tempDir, "input_image.tensor")
                saveFloatArrayToFile(inputImageFile, imageData, longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong()))
                scaledBitmap.recycle()
                Part12Saved(inputImageFile, mergedSize1x, mergedSize05x, System.currentTimeMillis() - part12Start)
            }
            System.gc(); System.runFinalization()
            Log.d(TAG, "FP16 Part 1+2 intermediates on disk. Memory: ${getMemoryInfo()}")

            // Part 3
            progressCallback?.invoke(0.50f, "FP16 Part 3: Image encoder...")
            val part3File = findFile(SPLIT_PART3) ?: return null
            val part3Start = System.currentTimeMillis()
            run {
                val (imageDataP3, _) = loadFloatArrayFromFile(part12Saved.inputImageFile)
                val module3 = Module.load(part3File.absolutePath, Module.LOAD_MODE_MMAP)
                val inputTensor = createHalfTensor(imageDataP3, longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong()))
                val imageTokens = getOutputAsFloatArray(module3.forward(EValue.from(inputTensor))[0].toTensor())
                saveFloatArrayToFile(File(tempDir, "image_tokens.tensor"), imageTokens, longArrayOf(1, 577, 1024))
                module3.destroy()
            }
            System.gc(); System.runFinalization()
            val part3Time = System.currentTimeMillis() - part3Start
            Log.d(TAG, "FP16 Part 3 done in ${part3Time}ms Memory: ${getMemoryInfo()}")

            // Part 4: chunked (4a_512 FP16 + 4a_65 FP16 + 4b FP32) or single FP32 fallback
            progressCallback?.invoke(0.60f, "Part 4: Decoder...")
            val shapeLatent = longArrayOf(1, FEATURE_DIM.toLong(), part12Saved.mergedSize1x.toLong(), part12Saved.mergedSize1x.toLong())
            val shapeX1 = longArrayOf(1, FEATURE_DIM.toLong(), part12Saved.mergedSize05x.toLong(), part12Saved.mergedSize05x.toLong())
            val shapeX2 = longArrayOf(1, FEATURE_DIM.toLong(), SPATIAL_SIZE.toLong(), SPATIAL_SIZE.toLong())

            fun loadFp32Input(file: File, shape: LongArray): EValue {
                val (data, _) = loadFloatArrayFromFile(file)
                return EValue.from(Tensor.fromBlob(data, shape))
            }

            val part4Start = System.currentTimeMillis()
            val part4Output: Array<EValue>
            var part4ModuleToDestroy: Module? = null
            val useChunkedPart4 = isChunkedPart4Available()
            Log.d(TAG, "Part 4: chunked=$useChunkedPart4. Memory: ${getMemoryInfo()}")

            if (useChunkedPart4) {
                // Chunked: 4a_512 (FP16) → 4a_65 (FP16) → concat → 4b (FP32 decoder)
                val part4a512File = findFile(SPLIT_PART4A_CHUNK_512)!!
                val part4a65File = findFile(SPLIT_PART4A_CHUNK_65)!!
                val part4bFile = findFile(SPLIT_PART4B)!!

                val (imageTokensFull, _) = loadFloatArrayFromFile(File(tempDir, "image_tokens.tensor"))

                // 4a: 512-token chunk (FP16)
                val chunk512Input = FloatArray(CHUNK_LEN_FIRST * FEATURE_DIM)
                System.arraycopy(imageTokensFull, 0, chunk512Input, 0, chunk512Input.size)
                System.gc()
                var module4a = Module.load(part4a512File.absolutePath, Module.LOAD_MODE_MMAP)
                val chunk512Tensor = createHalfTensor(chunk512Input, longArrayOf(1, CHUNK_LEN_FIRST.toLong(), FEATURE_DIM.toLong()))
                val out512 = getOutputAsFloatArray(module4a.forward(EValue.from(chunk512Tensor))[0].toTensor())
                module4a.destroy()
                System.gc(); System.runFinalization()
                Log.d(TAG, "Part 4a (512) done. Memory: ${getMemoryInfo()}")

                // 4a: 65-token chunk (FP16)
                val chunk65Input = FloatArray(CHUNK_LEN_LAST * FEATURE_DIM)
                System.arraycopy(imageTokensFull, CHUNK_LEN_FIRST * FEATURE_DIM, chunk65Input, 0, chunk65Input.size)
                System.gc()
                module4a = Module.load(part4a65File.absolutePath, Module.LOAD_MODE_MMAP)
                val chunk65Tensor = createHalfTensor(chunk65Input, longArrayOf(1, CHUNK_LEN_LAST.toLong(), FEATURE_DIM.toLong()))
                val out65 = getOutputAsFloatArray(module4a.forward(EValue.from(chunk65Tensor))[0].toTensor())
                module4a.destroy()
                System.gc(); System.runFinalization()
                Log.d(TAG, "Part 4a (65) done. Memory: ${getMemoryInfo()}")

                // Concat token outputs
                val tokensAfterBlocks = FloatArray(IMAGE_TOKENS_SEQ_LEN * FEATURE_DIM)
                System.arraycopy(out512, 0, tokensAfterBlocks, 0, out512.size)
                System.arraycopy(out65, 0, tokensAfterBlocks, out512.size, out65.size)
                System.gc()

                // 4b: decoder (FP32)
                Log.d(TAG, "Part 4b (decoder FP32) starting. Memory: ${getMemoryInfo()}")
                val module4b = Module.load(part4bFile.absolutePath, Module.LOAD_MODE_MMAP)
                val ev1Tokens = EValue.from(Tensor.fromBlob(tokensAfterBlocks, longArrayOf(1, IMAGE_TOKENS_SEQ_LEN.toLong(), FEATURE_DIM.toLong())))
                System.gc()
                val ev2Image = loadFp32Input(part12Saved.inputImageFile, longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong()))
                System.gc()
                val ev3 = loadFp32Input(File(tempDir, "latent0.tensor"), shapeLatent)
                System.gc()
                val ev4 = loadFp32Input(File(tempDir, "latent1.tensor"), shapeLatent)
                System.gc()
                val ev5 = loadFp32Input(File(tempDir, "x0Feat.tensor"), shapeLatent)
                System.gc()
                val ev6 = loadFp32Input(File(tempDir, "x1Feat.tensor"), shapeX1)
                System.gc()
                val ev7 = loadFp32Input(File(tempDir, "x2.tensor"), shapeX2)
                System.gc(); System.runFinalization()

                part4Output = try {
                    val out = module4b.forward(ev1Tokens, ev2Image, ev3, ev4, ev5, ev6, ev7)
                    Log.d(TAG, "Part 4b forward returned in ${System.currentTimeMillis() - part4Start}ms")
                    out
                } catch (e: OutOfMemoryError) {
                    Log.e(TAG, "Part 4b: OOM during decoder forward.", e)
                    module4b.destroy(); System.gc()
                    progressCallback?.invoke(0f, "Out of memory during decoder. Close other apps or use 6GB+ RAM device.")
                    return null
                }
                part4ModuleToDestroy = module4b
            } else {
                // Single Part 4 fallback (FP32)
                val part4File = findFile(SPLIT_PART4) ?: return null
                Log.d(TAG, "Part 4: single FP32 path. Memory: ${getMemoryInfo()}")
                System.gc(); System.runFinalization()
                val module4 = Module.load(part4File.absolutePath, Module.LOAD_MODE_MMAP)
                System.gc()
                val ev1 = loadFp32Input(part12Saved.inputImageFile, longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong()))
                System.gc()
                val ev2 = loadFp32Input(File(tempDir, "image_tokens.tensor"), longArrayOf(1, 577, 1024))
                System.gc()
                val ev3 = loadFp32Input(File(tempDir, "latent0.tensor"), shapeLatent)
                System.gc()
                val ev4 = loadFp32Input(File(tempDir, "latent1.tensor"), shapeLatent)
                System.gc()
                val ev5 = loadFp32Input(File(tempDir, "x0Feat.tensor"), shapeLatent)
                System.gc()
                val ev6 = loadFp32Input(File(tempDir, "x1Feat.tensor"), shapeX1)
                System.gc()
                val ev7 = loadFp32Input(File(tempDir, "x2.tensor"), shapeX2)
                System.gc(); System.runFinalization()

                Log.d(TAG, "Part 4: forward starting. Memory: ${getMemoryInfo()}")
                part4Output = try {
                    val out = module4.forward(ev1, ev2, ev3, ev4, ev5, ev6, ev7)
                    Log.d(TAG, "Part 4: forward returned in ${System.currentTimeMillis() - part4Start}ms")
                    out
                } catch (e: OutOfMemoryError) {
                    Log.e(TAG, "Part 4: OOM during decoder forward.", e)
                    module4.destroy(); System.gc()
                    progressCallback?.invoke(0f, "Out of memory during decoder.")
                    return null
                }
                part4ModuleToDestroy = module4
            }

            val part4Time = System.currentTimeMillis() - part4Start
            Log.d(TAG, "Part 4 done in ${part4Time}ms Memory: ${getMemoryInfo()}")

            val outputTensor = part4Output[0].toTensor()
            val outputBuffer = getTensorDataAsFloatBuffer(outputTensor)
            val gaussianCount = (outputTensor.numel() / PARAMS_PER_GAUSSIAN).toInt().coerceAtLeast(0)
            progressCallback?.invoke(0.80f, "Writing PLY ($gaussianCount Gaussians)...")
            val result = writeStreamingPlyPackedFromBuffer(gaussianCount, outputBuffer, progressCallback)
            outputBuffer.clear()
            part4ModuleToDestroy?.destroy()
            tempDir.listFiles()?.forEach { it.delete() }
            System.gc(); System.runFinalization()

            val elapsed = System.currentTimeMillis() - startTime
            Log.d(TAG, "inferSplitMode FP16 SUCCESS totalMs=$elapsed gaussianCount=$gaussianCount")
            Log.d(TAG, "  Breakdown: P1+P2=${part12Saved.elapsedMs}ms P3=${part3Time}ms P4=${part4Time}ms")

            progressCallback?.invoke(1.0f, "Done!")
            return result

        } catch (e: Exception) {
            Log.e(TAG, "FP16 split inference failed after ${System.currentTimeMillis() - startTime}ms", e)
            progressCallback?.invoke(0f, "Error: ${e.message}")
            return null
        }
    }

    // ---- PLY writing (same as ExecutorchSharp) ----

    private fun writeStreamingPlyPackedFromBuffer(
        gaussianCount: Int, params: FloatBuffer,
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

        Log.d(TAG, "PLY writer: reusable buffers (plyBatch=${BYTES_PER_VERTEX * PLY_BATCH_SIZE / 1024}KB, zeroSH=180B direct)")
        val batchLoopStart: Long
        var totalWriteNs = 0L
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
            val progressEvery = max(1, gaussianCount / 10)
            batchLoopStart = System.currentTimeMillis()
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
                    batchBuffer.putFloat(x); batchBuffer.putFloat(y); batchBuffer.putFloat(z)
                    batchBuffer.putFloat(0f); batchBuffer.putFloat(0f); batchBuffer.putFloat(0f)
                    val r = params.get(offset + 11).coerceIn(0f, 1f)
                    val g = params.get(offset + 12).coerceIn(0f, 1f)
                    val b = params.get(offset + 13).coerceIn(0f, 1f)
                    batchBuffer.putFloat((r - 0.5f) / SH_C0)
                    batchBuffer.putFloat((g - 0.5f) / SH_C0)
                    batchBuffer.putFloat((b - 0.5f) / SH_C0)
                    zeroSHBuffer.clear()
                    batchBuffer.put(zeroSHBuffer)
                    val rawOpacity = params.get(offset + 3).coerceIn(0f, 1f)
                    val lutIndex = (rawOpacity * lutScale).toInt().coerceIn(0, LOGIT_LUT_SIZE - 1)
                    batchBuffer.putFloat(LOGIT_LUT[lutIndex])
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
                val writeStart = System.nanoTime()
                while (batchBuffer.hasRemaining()) channel.write(batchBuffer)
                totalWriteNs += System.nanoTime() - writeStart
                batchBuffer.clear()
                processed += currentBatch
                if (processed % progressEvery == 0 || processed == gaussianCount) {
                    progressCallback?.invoke(0.75f + (processed.toFloat() / gaussianCount) * 0.20f, "Writing PLY ($processed/$gaussianCount)...")
                }
            }
        }
        val batchLoopTime = System.currentTimeMillis() - batchLoopStart
        val totalWriteMs = totalWriteNs / 1_000_000
        val plyBytes = gaussianCount.toLong() * BYTES_PER_VERTEX
        val throughputMBps = if (batchLoopTime > 0) plyBytes * 1000.0 / batchLoopTime / 1024 / 1024 else 0.0
        Log.d(TAG, "PLY batch loop: ${batchLoopTime}ms total (I/O write=${totalWriteMs}ms, compute=${batchLoopTime - totalWriteMs}ms)")
        Log.d(TAG, "PLY throughput: ${plyBytes / 1024 / 1024}MB @ ${String.format("%.1f", throughputMBps)}MB/s, batches=${(gaussianCount + PLY_BATCH_SIZE - 1) / PLY_BATCH_SIZE}")

        plyFile.copyTo(classicPlyFile, overwrite = true)
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

    fun release() {
        isInitialized = false
        Log.d(TAG, "ExecutorchFp16Sharp released")
    }
}
