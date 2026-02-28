package com.furnit.android.services

import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import com.furnit.android.utils.LogUtil
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.withContext
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.Date
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Native .pt SHARP inference via PyTorch Mobile (LibTorch) runtime.
 *
 * PREFERS SPLIT MODE (avoids OOM): sharp_scripted_part1.ptl ... part4.ptl
 * Each part ~500-800MB; load one at a time. Full 2.5GB model crashes on load.
 *
 * Split pipeline: Part1 (35 patches) -> Part2 (35) -> merge -> Part3 (1) -> Part4 (1)
 * Same architecture as ExecuTorch/LiteRT split.
 *
 * Fallback: sharp_scripted.ptl (full model) - may OOM on devices.
 */
class NativePtSharp private constructor(private val context: Context) {

    companion object {
        private const val TAG = "NativePtSharp"
        private const val IMAGE_SIZE = 1536
        private const val PARAMS_PER_GAUSSIAN = 14
        private const val SH_C0 = 0.28209479177387814f
        private const val BYTES_PER_VERTEX = 62 * 4
        private const val LOGIT_LUT_SIZE = 1024

        private val LOGIT_LUT = FloatArray(LOGIT_LUT_SIZE) { i ->
            val p = (i.toFloat() / (LOGIT_LUT_SIZE - 1)).coerceIn(1e-4f, 1f - 1e-4f)
            ln(p / (1f - p))
        }

        private val FULL_MODEL_FILES = listOf("sharp_scripted.ptl", "sharp_mobile.ptl")
        private const val SPLIT_PART1 = "sharp_scripted_part1.ptl"
        private const val SPLIT_PART2 = "sharp_scripted_part2.ptl"
        private const val SPLIT_PART3 = "sharp_scripted_part3.ptl"
        private const val SPLIT_PART4 = "sharp_scripted_part4.ptl"
        private val SPLIT_FILENAMES = listOf(SPLIT_PART1, SPLIT_PART2, SPLIT_PART3, SPLIT_PART4)

        private const val PATCH_SIZE = 384
        private const val FEATURE_DIM = 1024
        private const val SPATIAL_SIZE = 24
        private const val GRID_1X = 5
        private const val GRID_05X = 3
        private const val PADDING_1X = 3
        private const val PADDING_05X = 6
        private const val TOTAL_PATCHES = 35
        private const val PATCHES_1X = 25
        private const val PATCHES_05X = 9

        /** Expected merged sizes from Python merge_patches_from_list (RAG/docs). */
        private const val EXPECTED_MERGED_SIZE_1X = 96
        private const val EXPECTED_MERGED_SIZE_05X = 48

        /** Use parallel patch preparation (extract + preprocess). Forward remains sequential for model thread-safety. */
        private const val USE_PARALLEL_PATCH_PREP = true

        /** Part 2 forward: must be sequential - PyTorch Mobile forward() is not thread-safe. */
        private const val USE_PARALLEL_PART2 = false

        @Volatile
        private var instance: NativePtSharp? = null

        fun getInstance(context: Context): NativePtSharp {
            return instance ?: synchronized(this) {
                instance ?: NativePtSharp(context.applicationContext).also { instance = it }
            }
        }
    }

    private var module: org.pytorch.Module? = null
    private var isInitialized = false
    private var useSplitMode = false

    private val internalModelsDir: File by lazy {
        File(context.filesDir, "models")
    }

    private val externalModelsDir: File? by lazy {
        context.getExternalFilesDir("models")
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
                try {
                    internalModelsDir.mkdirs()
                    external.copyTo(internal, overwrite = true)
                    LogUtil.d(TAG, "findFile: Copied $filename to internal storage from ${external.absolutePath}")
                    return internal
                } catch (e: Exception) {
                    LogUtil.d(TAG, "findFile: copy failed, using external path=${external.absolutePath}")
                    return external
                }
            }
        }
        val tmp = File("/data/local/tmp/furnit/", filename)
        if (tmp.exists() && tmp.length() > 0) {
            LogUtil.d(TAG, "findFile: $filename found tmp path=${tmp.absolutePath} size=${tmp.length()}")
            return tmp
        }
        LogUtil.w(TAG, "findFile: $filename NOT FOUND")
        return null
    }

    private fun isSplitModelReady(): Boolean = SPLIT_FILENAMES.all { findFile(it) != null }

    private fun ensureModelInInternalStorage(): File? {
        for (name in FULL_MODEL_FILES) {
            val f = findFile(name)
            if (f != null) return f
        }
        return null
    }

    private fun findModelFile(): File? = ensureModelInInternalStorage()

    fun isModelReady(): Boolean = isSplitModelReady() || findModelFile() != null

    fun initialize(): Boolean {
        LogUtil.d(TAG, "initialize ENTER. Memory: ${getMemoryInfo()}")
        if (isSplitModelReady()) {
            useSplitMode = true
            module = null
            isInitialized = true
            LogUtil.d(TAG, "initialize: Native .pt SPLIT mode: 4 parts (~500-800MB each, load on demand)")
            SPLIT_FILENAMES.forEach { fn -> LogUtil.d(TAG, "initialize: split part ${findFile(fn)?.absolutePath ?: "MISSING"}") }
            return true
        }
        val modelFile = findModelFile()
        if (modelFile == null) {
            LogUtil.e(TAG, "Model not found. Push sharp_scripted_part1-4.ptl (split) or sharp_scripted.ptl (full).")
            return false
        }
        val sizeMB = modelFile.length() / 1024 / 1024
        LogUtil.d(TAG, "Native .pt FULL model: ${modelFile.name} (${sizeMB}MB) - may OOM")

        try {
            LogUtil.d(TAG, "Native .pt: LiteModuleLoader.load...")
            val loadStart = System.currentTimeMillis()
            module = LiteModuleLoader.load(modelFile.absolutePath)
            useSplitMode = false
            isInitialized = true
            LogUtil.d(TAG, "Native .pt full initialized in ${System.currentTimeMillis() - loadStart}ms")
            return true
        } catch (e: OutOfMemoryError) {
            LogUtil.e(TAG, "Native .pt OOM: Use split model (export_sharp_torchscript_split.py)", e)
            return false
        } catch (e: Exception) {
            LogUtil.e(TAG, "Native .pt init failed: ${e.javaClass.simpleName} ${e.message}", e)
            return false
        }
    }

    suspend fun inferStreaming(
        bitmap: Bitmap,
        progressCallback: ((Float, String) -> Unit)? = null,
        isCancelled: () -> Boolean = { false }
    ): StreamingResult? = withContext(Dispatchers.IO) {
        LogUtil.d(TAG, "inferStreaming ENTER bitmap=${bitmap.width}x${bitmap.height} useSplitMode=$useSplitMode")
        if (!isInitialized) {
            LogUtil.e(TAG, "inferStreaming: Not initialized")
            return@withContext null
        }
        if (useSplitMode) {
            LogUtil.d(TAG, "inferStreaming: delegating to inferStreamingSplit")
            return@withContext inferStreamingSplit(bitmap, progressCallback, isCancelled)
        }

        try {
            LogUtil.d(TAG, "inferStreaming: FULL model path, preprocessing...")
            progressCallback?.invoke(0.05f, "Preprocessing image...")
            val scaledBitmap = SharpImagePreprocessor.resizeForSharp(bitmap)
            val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
            scaledBitmap.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
            scaledBitmap.recycle()

            val floatData = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE)
            val channelSize = IMAGE_SIZE * IMAGE_SIZE
            val inv255 = 1f / 255f
            for (i in pixels.indices) {
                floatData[i] = ((pixels[i] shr 16) and 0xFF) * inv255
                floatData[channelSize + i] = ((pixels[i] shr 8) and 0xFF) * inv255
                floatData[2 * channelSize + i] = (pixels[i] and 0xFF) * inv255
            }

            val inputTensor = Tensor.fromBlob(
                floatData,
                longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong())
            )

            progressCallback?.invoke(0.2f, "Running Native .pt inference (LibTorch)...")
            val inferStart = System.currentTimeMillis()
            val output = module!!.forward(IValue.from(inputTensor)).toTensor()
            val inferTime = System.currentTimeMillis() - inferStart
            LogUtil.d(TAG, "Native .pt inference: ${inferTime}ms")

            val outputData = output.dataAsFloatArray
            val gaussianCount = outputData.size / PARAMS_PER_GAUSSIAN
            LogUtil.d(TAG, "Produced $gaussianCount Gaussians (Native .pt)")

            progressCallback?.invoke(0.8f, "Writing PLY ($gaussianCount Gaussians)...")
            val roomsDir = File(context.filesDir, "sharp_rooms")
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val roomFolder = File(roomsDir, "room_$timestamp")
            roomFolder.mkdirs()

            val plyFile = File(roomFolder, "room.ply")
            val classicPlyFile = File(roomFolder, "room_classic.ply")

            writePly(plyFile, outputData, gaussianCount)
            plyFile.copyTo(classicPlyFile, overwrite = true)

            var minX = Float.MAX_VALUE
            var maxX = -Float.MAX_VALUE
            var minY = Float.MAX_VALUE
            var maxY = -Float.MAX_VALUE
            var minZ = Float.MAX_VALUE
            var maxZ = -Float.MAX_VALUE
            for (i in 0 until gaussianCount) {
                val off = i * PARAMS_PER_GAUSSIAN
                val x = outputData[off]
                val y = outputData[off + 1]
                val z = outputData[off + 2]
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
                if (z < minZ) minZ = z
                if (z > maxZ) maxZ = z
            }

            val roomWidth = maxX - minX
            val roomHeight = maxY - minY
            val roomDepth = maxZ - minZ
            LogUtil.d(TAG, "Native .pt done: $gaussianCount Gaussians, room ${roomWidth}m x ${roomHeight}m x ${roomDepth}m")

            progressCallback?.invoke(1.0f, "Done!")
            return@withContext StreamingResult(
                plyFile = plyFile,
                classicPlyFile = classicPlyFile,
                gaussianCount = gaussianCount,
                roomWidth = roomWidth,
                roomHeight = roomHeight,
                roomDepth = roomDepth
            )
        } catch (e: Exception) {
            LogUtil.e(TAG, "Native .pt inference failed", e)
            progressCallback?.invoke(0f, "Error: ${e.message}")
            return@withContext null
        }
    }

    private fun getMemoryInfo(): String {
        val runtime = Runtime.getRuntime()
        val usedMB = (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024
        val maxMB = runtime.maxMemory() / 1024 / 1024
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        val availSysMB = memInfo.availMem / 1024 / 1024
        return "JVM: ${usedMB}/${maxMB}MB, SysAvail: ${availSysMB}MB"
    }

    private fun getMergedSize(gridSize: Int, patchSpatial: Int, padding: Int): Int {
        val patchContrib = patchSpatial - 2 * padding
        return patchSpatial + (gridSize - 1) * patchContrib
    }

    private data class FeatureStats(
        val min: Float,
        val max: Float,
        val mean: Float,
        val hasNan: Boolean,
        val hasInf: Boolean,
        val sampleCount: Int
    )

    private fun computeFeatureStats(data: FloatArray, name: String): FeatureStats {
        if (data.isEmpty()) return FeatureStats(0f, 0f, 0f, false, false, 0)
        var minVal = Float.MAX_VALUE
        var maxVal = -Float.MAX_VALUE
        var sum = 0.0
        var hasNan = false
        var hasInf = false
        for (v in data) {
            if (v != v) hasNan = true
            if (v.isInfinite()) hasInf = true
            if (!hasNan && !hasInf) {
                if (v < minVal) minVal = v
                if (v > maxVal) maxVal = v
                sum += v
            }
        }
        val mean = if (data.isNotEmpty()) (sum / data.size).toFloat() else 0f
        LogUtil.d(TAG, "Quality[$name]: min=$minVal max=$maxVal mean=$mean hasNan=$hasNan hasInf=$hasInf n=${data.size}")
        return FeatureStats(minVal, maxVal, mean, hasNan, hasInf, data.size)
    }

    private fun validateGaussianOutput(params: FloatArray, gaussianCount: Int): Boolean {
        if (gaussianCount <= 0 || params.size < gaussianCount * PARAMS_PER_GAUSSIAN) return false
        var nanCount = 0
        var infCount = 0
        var outOfBoundsPos = 0
        for (i in 0 until gaussianCount) {
            val off = i * PARAMS_PER_GAUSSIAN
            for (k in 0 until 6) {
                val v = params[off + k]
                if (v != v) nanCount++
                if (v.isInfinite()) infCount++
            }
            val x = params[off]; val y = params[off + 1]; val z = params[off + 2]
            if (x.isNaN() || y.isNaN() || z.isNaN() || x.isInfinite() || y.isInfinite() || z.isInfinite()) {
                nanCount++; infCount++
            }
            if (kotlin.math.abs(x) > 100f || kotlin.math.abs(y) > 100f || kotlin.math.abs(z) > 100f) outOfBoundsPos++
        }
        val ok = nanCount == 0 && infCount == 0
        LogUtil.d(TAG, "Quality[Gaussians]: gaussianCount=$gaussianCount nanCount=$nanCount infCount=$infCount outOfBoundsPos=$outOfBoundsPos ok=$ok")
        return ok
    }

    private fun validateMergeSizes(mergedSize1x: Int, mergedSize05x: Int): Boolean {
        val ok1 = mergedSize1x == EXPECTED_MERGED_SIZE_1X
        val ok2 = mergedSize05x == EXPECTED_MERGED_SIZE_05X
        if (!ok1 || !ok2) {
            LogUtil.e(TAG, "Quality[Merge]: mergedSize1x=$mergedSize1x (expected $EXPECTED_MERGED_SIZE_1X) mergedSize05x=$mergedSize05x (expected $EXPECTED_MERGED_SIZE_05X)")
        } else {
            LogUtil.d(TAG, "Quality[Merge]: mergedSize1x=$mergedSize1x mergedSize05x=$mergedSize05x OK")
        }
        return ok1 && ok2
    }

    private fun preprocessPatch(bitmap: Bitmap): FloatArray {
        val out = FloatArray(3 * bitmap.width * bitmap.height)
        preprocessPatchInto(bitmap, out)
        return out
    }

    /**
     * Prepares a single patch buffer for the given index. Indices 0-24: 1x grid, 25-33: 0.5x, 34: quarter.
     * Safe to call from multiple threads (each reads from its own bitmap region).
     */
    private fun preparePatchAtIndex(
        scaledBitmap: Bitmap,
        halfBitmap: Bitmap,
        quarterBitmap: Bitmap,
        index: Int,
        stride1x: Int,
        stride05x: Int
    ): FloatArray {
        val out = FloatArray(3 * PATCH_SIZE * PATCH_SIZE)
        val patchBitmap = when {
            index < PATCHES_1X -> {
                val i = index / GRID_1X
                val j = index % GRID_1X
                Bitmap.createBitmap(scaledBitmap, j * stride1x, i * stride1x, PATCH_SIZE, PATCH_SIZE)
            }
            index < PATCHES_1X + PATCHES_05X -> {
                val local = index - PATCHES_1X
                val i = local / GRID_05X
                val j = local % GRID_05X
                Bitmap.createBitmap(halfBitmap, j * stride05x, i * stride05x, PATCH_SIZE, PATCH_SIZE)
            }
            else -> Bitmap.createScaledBitmap(quarterBitmap, PATCH_SIZE, PATCH_SIZE, true)
        }
        preprocessPatchInto(patchBitmap, out)
        patchBitmap.recycle()
        return out
    }

    private fun preprocessPatchInto(bitmap: Bitmap, out: FloatArray) {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        val channelSize = width * height
        val inv255 = 1f / 255f
        for (i in pixels.indices) {
            out[i] = ((pixels[i] shr 16) and 0xFF) * inv255
            out[channelSize + i] = ((pixels[i] shr 8) and 0xFF) * inv255
            out[2 * channelSize + i] = (pixels[i] and 0xFF) * inv255
        }
    }

    private suspend fun inferStreamingSplit(
        bitmap: Bitmap,
        progressCallback: ((Float, String) -> Unit)?,
        isCancelled: () -> Boolean = { false }
    ): StreamingResult? {
        fun mergeOnePatchInto(
            output: FloatArray,
            outSize: Int,
            patch: FloatArray,
            gridI: Int,
            gridJ: Int,
            gridSize: Int,
            padding: Int
        ) {
            val patchH = SPATIAL_SIZE
            val patchW = SPATIAL_SIZE
            val srcY0 = if (gridJ == 0) 0 else padding
            val srcY1 = if (gridJ == gridSize - 1) patchH else (patchH - padding)
            val copyH = srcY1 - srcY0
            val srcX0 = if (gridI == 0) 0 else padding
            val srcX1 = if (gridI == gridSize - 1) patchW else (patchW - padding)
            val copyW = srcX1 - srcX0
            // Placement matches Python merge_patches_from_list: first and last patches contribute
            // (patchW-padding), inner patches (patchW-2*padding). Total = 96 for 5x5, padding=3.
            val firstContrib = patchW - padding
            val innerContrib = patchW - 2 * padding
            val outX = if (gridI == 0) 0 else firstContrib + (gridI - 1) * innerContrib
            val outY = if (gridJ == 0) 0 else firstContrib + (gridJ - 1) * innerContrib
            for (c in 0 until FEATURE_DIM) {
                val srcBase = c * patchH * patchW
                val dstBase = c * outSize * outSize
                for (dy in 0 until copyH) {
                    val srcOff = srcBase + (srcY0 + dy) * patchW + srcX0
                    val dstOff = dstBase + (outY + dy) * outSize + outX
                    System.arraycopy(patch, srcOff, output, dstOff, copyW)
                }
            }
        }

        val tempSpatial = FloatArray(FEATURE_DIM * SPATIAL_SIZE * SPATIAL_SIZE)
        fun reshapeTokensToSpatialInto(tokens: FloatArray, spatialOut: FloatArray) {
            val hw = SPATIAL_SIZE * SPATIAL_SIZE
            for (dstPix in 0 until hw) {
                val seqIdx = 1 + dstPix
                val srcOff = seqIdx * FEATURE_DIM
                for (c in 0 until FEATURE_DIM) {
                    spatialOut[c * hw + dstPix] = tokens[srcOff + c]
                }
            }
        }

        val startTime = System.currentTimeMillis()
        LogUtil.d(TAG, "inferStreamingSplit ENTER. Memory: ${getMemoryInfo()} bitmap=${bitmap.width}x${bitmap.height}")
        try {
            if (isCancelled()) {
                LogUtil.d(TAG, "inferStreamingSplit CANCELLED before preprocessing")
                return null
            }
            progressCallback?.invoke(0.02f, "Preprocessing...")
            val preprocessStart = System.currentTimeMillis()
            val scaledBitmap = SharpImagePreprocessor.resizeForSharp(bitmap)
            val halfSize = IMAGE_SIZE / 2
            val halfBitmap = Bitmap.createScaledBitmap(scaledBitmap, halfSize, halfSize, true)
            val quarterBitmap = Bitmap.createScaledBitmap(scaledBitmap, PATCH_SIZE, PATCH_SIZE, true)
            LogUtil.d(TAG, "inferStreamingSplit: preprocessing done in ${System.currentTimeMillis() - preprocessStart}ms scaled=${scaledBitmap.width}x${scaledBitmap.height} half=${halfSize} quarter=${PATCH_SIZE}")

            val mergedSize1x = getMergedSize(GRID_1X, SPATIAL_SIZE, PADDING_1X)
            val mergedSize05x = getMergedSize(GRID_05X, SPATIAL_SIZE, PADDING_05X)
            validateMergeSizes(mergedSize1x, mergedSize05x)
            var latent0 = FloatArray(FEATURE_DIM * mergedSize1x * mergedSize1x)
            var latent1 = FloatArray(FEATURE_DIM * mergedSize1x * mergedSize1x)
            var x0Feat = FloatArray(FEATURE_DIM * mergedSize1x * mergedSize1x)
            var x1Feat = FloatArray(FEATURE_DIM * mergedSize05x * mergedSize05x)
            var x2Feat: FloatArray? = null

            val patchShape = longArrayOf(1, 3, PATCH_SIZE.toLong(), PATCH_SIZE.toLong())
            val stride1x = (IMAGE_SIZE - PATCH_SIZE) / 4
            val stride05x = (halfSize - PATCH_SIZE) / 2
            LogUtil.d(TAG, "inferStreamingSplit: stride1x=$stride1x stride05x=$stride05x patchShape=[1,3,$PATCH_SIZE,$PATCH_SIZE] merged1x=$mergedSize1x merged05x=$mergedSize05x")

            val prePreparedPatches: List<FloatArray>? = if (USE_PARALLEL_PATCH_PREP) {
                progressCallback?.invoke(0.03f, "Part 1: Preparing patches in parallel...")
                val prepStart = System.currentTimeMillis()
                coroutineScope {
                    (0 until TOTAL_PATCHES).map { idx ->
                        async(Dispatchers.Default) {
                            preparePatchAtIndex(scaledBitmap, halfBitmap, quarterBitmap, idx, stride1x, stride05x)
                        }
                    }.awaitAll()
                }.also { LogUtil.d(TAG, "Prepared $TOTAL_PATCHES patches in parallel in ${System.currentTimeMillis() - prepStart}ms") }
            } else null

            if (USE_PARALLEL_PATCH_PREP) {
                halfBitmap.recycle()
                quarterBitmap.recycle()
            }

            if (isCancelled()) {
                LogUtil.d(TAG, "inferStreamingSplit CANCELLED before Part 1")
                return null
            }
            // Part 1: Patch Encoder A
            progressCallback?.invoke(0.05f, "Part 1: Loading encoder A...")
            val part1File = findFile(SPLIT_PART1) ?: return null
            LogUtil.d(TAG, "inferStreamingSplit: Part1 load from ${part1File.absolutePath} size=${part1File.length() / 1024 / 1024}MB")
            var ptModule = LiteModuleLoader.load(part1File.absolutePath)
            val reusablePatchBuffer = FloatArray(3 * PATCH_SIZE * PATCH_SIZE)
            val warmupStart = System.currentTimeMillis()
            ptModule.forward(IValue.from(Tensor.fromBlob(reusablePatchBuffer, patchShape)))
            LogUtil.d(TAG, "inferStreamingSplit: Part1 warmup done in ${System.currentTimeMillis() - warmupStart}ms")

            val allTokens = ArrayList<FloatArray>(TOTAL_PATCHES)
            var patchCount = 0
            val part1FwdStart = System.currentTimeMillis()

            for (idx in 0 until TOTAL_PATCHES) {
                val patchData = when {
                    prePreparedPatches != null -> prePreparedPatches[idx]
                    idx < PATCHES_1X -> {
                        val patchBitmap = Bitmap.createBitmap(scaledBitmap, (idx % GRID_1X) * stride1x, (idx / GRID_1X) * stride1x, PATCH_SIZE, PATCH_SIZE)
                        preprocessPatchInto(patchBitmap, reusablePatchBuffer)
                        patchBitmap.recycle()
                        reusablePatchBuffer
                    }
                    idx < PATCHES_1X + PATCHES_05X -> {
                        val local = idx - PATCHES_1X
                        val patchBitmap = Bitmap.createBitmap(halfBitmap, local % GRID_05X * stride05x, local / GRID_05X * stride05x, PATCH_SIZE, PATCH_SIZE)
                        preprocessPatchInto(patchBitmap, reusablePatchBuffer)
                        patchBitmap.recycle()
                        reusablePatchBuffer
                    }
                    else -> {
                        preprocessPatchInto(quarterBitmap, reusablePatchBuffer)
                        reusablePatchBuffer
                    }
                }
                val inputTensor = Tensor.fromBlob(patchData, patchShape)
                val out = ptModule.forward(IValue.from(inputTensor))
                val tuple = out.toTuple()
                val tokens = tuple[0].toTensor().dataAsFloatArray
                val block5 = tuple[1].toTensor().dataAsFloatArray
                if (idx < PATCHES_1X) {
                    allTokens.add(tokens)
                    reshapeTokensToSpatialInto(block5, tempSpatial)
                    mergeOnePatchInto(latent0, mergedSize1x, tempSpatial, idx % GRID_1X, idx / GRID_1X, GRID_1X, PADDING_1X)
                    reshapeTokensToSpatialInto(tokens, tempSpatial)
                    mergeOnePatchInto(latent1, mergedSize1x, tempSpatial, idx % GRID_1X, idx / GRID_1X, GRID_1X, PADDING_1X)
                } else {
                    allTokens.add(tokens)
                }
                patchCount++
                if (idx == 0 || idx == PATCHES_1X - 1 || idx == PATCHES_1X || idx == TOTAL_PATCHES - 1) {
                    LogUtil.d(TAG, "inferStreamingSplit: Part1 patch idx=$idx tokensLen=${tokens.size} block5Len=${block5.size} mergeInto=${if (idx < PATCHES_1X) "latent0+latent1" else "tokensOnly"}")
                }
                if (patchCount % 5 == 0) progressCallback?.invoke(0.05f + (patchCount.toFloat() / TOTAL_PATCHES) * 0.20f, "Part 1: Patch $patchCount/$TOTAL_PATCHES...")
            }

            LogUtil.d(TAG, "inferStreamingSplit: Part1 forward loop ${patchCount} patches in ${System.currentTimeMillis() - part1FwdStart}ms (avg ${(System.currentTimeMillis() - part1FwdStart) / patchCount}ms/patch)")

            if (!USE_PARALLEL_PATCH_PREP) {
                halfBitmap.recycle()
                quarterBitmap.recycle()
            }

            ptModule.destroy()
            ptModule = null
            System.gc()
            val part1Ms = System.currentTimeMillis() - startTime
            LogUtil.d(TAG, "Part 1 done: $patchCount tokens in ${part1Ms}ms. Memory: ${getMemoryInfo()}")
            computeFeatureStats(latent0, "latent0_after_part1")
            computeFeatureStats(latent1, "latent1_after_part1")

            if (isCancelled()) {
                LogUtil.d(TAG, "inferStreamingSplit CANCELLED before Part 2")
                return null
            }
            // Part 2: Patch Encoder B
            progressCallback?.invoke(0.30f, "Part 2: Loading encoder B...")
            LogUtil.d(TAG, "Part 2: before load. Memory: ${getMemoryInfo()}")
            val part2LoadStart = System.currentTimeMillis()
            val part2File = findFile(SPLIT_PART2) ?: return null
            LogUtil.d(TAG, "inferStreamingSplit: Part2 load from ${part2File.absolutePath} size=${part2File.length() / 1024 / 1024}MB")
            ptModule = LiteModuleLoader.load(part2File.absolutePath)
            LogUtil.d(TAG, "Part 2: load done in ${System.currentTimeMillis() - part2LoadStart}ms. Memory: ${getMemoryInfo()}")
            val part2Start = System.currentTimeMillis()
            val part2Features: List<FloatArray> = if (USE_PARALLEL_PART2) {
                val fwdStart = System.currentTimeMillis()
                val feats = coroutineScope {
                    (0 until patchCount).map { idx ->
                        async(Dispatchers.Default) {
                            val tokenTensor = Tensor.fromBlob(allTokens[idx], longArrayOf(1, 577, 1024))
                            ptModule.forward(IValue.from(tokenTensor)).toTensor().dataAsFloatArray
                        }
                    }.awaitAll()
                }
                LogUtil.d(TAG, "Part 2: parallel forward ${patchCount} patches in ${System.currentTimeMillis() - fwdStart}ms")
                feats
            } else {
                buildList {
                    for (idx in 0 until patchCount) {
                        val tokenTensor = Tensor.fromBlob(allTokens[idx], longArrayOf(1, 577, 1024))
                        val fwdStart = if (idx == 0) System.currentTimeMillis() else 0L
                        val out = ptModule.forward(IValue.from(tokenTensor))
                        if (idx == 0) LogUtil.d(TAG, "Part 2: first forward ${System.currentTimeMillis() - fwdStart}ms")
                        add(out.toTensor().dataAsFloatArray)
                        allTokens[idx] = FloatArray(0)
                        if (idx > 0 && idx % 10 == 0) System.gc()
                        if (idx % 5 == 0 || idx <= 1) LogUtil.d(TAG, "Part 2: patch $idx/$patchCount")
                        if (idx % 5 == 0) progressCallback?.invoke(0.30f + (idx.toFloat() / patchCount) * 0.15f, "Part 2: Feature $idx/$patchCount...")
                    }
                }
            }
            for (idx in 0 until patchCount) {
                val feat = part2Features[idx]
                when {
                    idx < PATCHES_1X -> mergeOnePatchInto(x0Feat, mergedSize1x, feat, idx % GRID_1X, idx / GRID_1X, GRID_1X, PADDING_1X)
                    idx < PATCHES_1X + PATCHES_05X -> {
                        val local = idx - PATCHES_1X
                        mergeOnePatchInto(x1Feat, mergedSize05x, feat, local % GRID_05X, local / GRID_05X, GRID_05X, PADDING_05X)
                    }
                    else -> x2Feat = feat
                }
                if (idx == 0 || idx == PATCHES_1X - 1 || idx == PATCHES_1X || idx == PATCHES_1X + PATCHES_05X - 1 || idx == patchCount - 1) {
                    LogUtil.d(TAG, "inferStreamingSplit: Part2 merge idx=$idx into=${when { idx < PATCHES_1X -> "x0Feat"; idx < PATCHES_1X + PATCHES_05X -> "x1Feat"; else -> "x2Feat" }} featLen=${feat.size}")
                }
                if (idx % 5 == 0 && !USE_PARALLEL_PART2) progressCallback?.invoke(0.30f + (idx.toFloat() / patchCount) * 0.15f, "Part 2: Merge $idx/$patchCount...")
            }
            LogUtil.d(TAG, "inferStreamingSplit: Part2 merge loop done, x2Feat=${x2Feat?.size ?: 0}")
            allTokens.clear()
            if (USE_PARALLEL_PART2) progressCallback?.invoke(0.45f, "Part 2: Merge done...")
            ptModule.destroy()
            ptModule = null
            allTokens.clear()
            System.gc()
            val part2Ms = System.currentTimeMillis() - part2Start
            LogUtil.d(TAG, "Part 2 done in ${part2Ms}ms. Memory: ${getMemoryInfo()}")
            computeFeatureStats(x0Feat, "x0Feat_after_part2")
            computeFeatureStats(x1Feat, "x1Feat_after_part2")

            val x2 = x2Feat ?: return null

            var imageData = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE)
            preprocessPatchInto(scaledBitmap, imageData)
            scaledBitmap.recycle()
            LogUtil.d(TAG, "inferStreamingSplit: imageData preprocessed ${IMAGE_SIZE}x${IMAGE_SIZE} len=${imageData.size} first3=${imageData.take(3)}")

            if (isCancelled()) {
                LogUtil.d(TAG, "inferStreamingSplit CANCELLED before Part 3")
                return null
            }
            // Part 3: Image Encoder A
            progressCallback?.invoke(0.50f, "Part 3: Image encoder...")
            LogUtil.d(TAG, "Part 3: before load. Memory: ${getMemoryInfo()}")
            val part3Start = System.currentTimeMillis()
            val part3File = findFile(SPLIT_PART3) ?: return null
            LogUtil.d(TAG, "inferStreamingSplit: Part3 load from ${part3File.absolutePath} size=${part3File.length() / 1024 / 1024}MB")
            ptModule = LiteModuleLoader.load(part3File.absolutePath)
            var imageTensor: Tensor? = Tensor.fromBlob(imageData, longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong()))
            val imageTokensOut = ptModule.forward(IValue.from(imageTensor))
            var imageTokens = imageTokensOut.toTensor().dataAsFloatArray
            LogUtil.d(TAG, "inferStreamingSplit: Part3 output imageTokens len=${imageTokens.size}")
            ptModule.destroy()
            ptModule = null
            System.gc()
            LogUtil.d(TAG, "Part 3 done in ${System.currentTimeMillis() - part3Start}ms. Memory: ${getMemoryInfo()}")

            if (isCancelled()) {
                LogUtil.d(TAG, "inferStreamingSplit CANCELLED before Part 4")
                return null
            }
            // Part 4: Decoder + Gaussians
            progressCallback?.invoke(0.60f, "Part 4: Decoder...")
            LogUtil.d(TAG, "Part 4: before load. Memory: ${getMemoryInfo()}")
            val part4LoadStart = System.currentTimeMillis()
            val part4File = findFile(SPLIT_PART4) ?: return null
            LogUtil.d(TAG, "inferStreamingSplit: Part4 load from ${part4File.absolutePath} size=${part4File.length() / 1024 / 1024}MB")
            ptModule = LiteModuleLoader.load(part4File.absolutePath)
            LogUtil.d(TAG, "Part 4: load done in ${System.currentTimeMillis() - part4LoadStart}ms. Input shapes: image[1,3,$IMAGE_SIZE,$IMAGE_SIZE] tokens[1,577,1024] latent0[1,$FEATURE_DIM,$mergedSize1x,$mergedSize1x] x0Feat x1Feat x2[1,$FEATURE_DIM,$SPATIAL_SIZE,$SPATIAL_SIZE]")
            val part4Start = System.currentTimeMillis()
            val part4Output = ptModule.forward(
                IValue.from(imageTensor!!),
                IValue.from(Tensor.fromBlob(imageTokens, longArrayOf(1, 577, 1024))),
                IValue.from(Tensor.fromBlob(latent0, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize1x.toLong(), mergedSize1x.toLong()))),
                IValue.from(Tensor.fromBlob(latent1, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize1x.toLong(), mergedSize1x.toLong()))),
                IValue.from(Tensor.fromBlob(x0Feat, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize1x.toLong(), mergedSize1x.toLong()))),
                IValue.from(Tensor.fromBlob(x1Feat, longArrayOf(1, FEATURE_DIM.toLong(), mergedSize05x.toLong(), mergedSize05x.toLong()))),
                IValue.from(Tensor.fromBlob(x2, longArrayOf(1, FEATURE_DIM.toLong(), SPATIAL_SIZE.toLong(), SPATIAL_SIZE.toLong())))
            )
            val packedOutput = part4Output.toTensor().dataAsFloatArray
            LogUtil.d(TAG, "inferStreamingSplit: Part4 output len=${packedOutput.size} paramsPerGaussian=$PARAMS_PER_GAUSSIAN first14=${packedOutput.take(14)}")
            ptModule.destroy()
            ptModule = null
            latent0 = FloatArray(0)
            latent1 = FloatArray(0)
            x0Feat = FloatArray(0)
            x1Feat = FloatArray(0)
            x2Feat = null
            imageData = FloatArray(0)
            imageTokens = FloatArray(0)
            imageTensor = null
            System.gc()
            LogUtil.d(TAG, "Part 4 done in ${System.currentTimeMillis() - part4Start}ms. Memory: ${getMemoryInfo()}")

            val gaussianCount = packedOutput.size / PARAMS_PER_GAUSSIAN
            LogUtil.d(TAG, "Native .pt SPLIT produced $gaussianCount Gaussians in ${System.currentTimeMillis() - startTime}ms")
            validateGaussianOutput(packedOutput, gaussianCount)

            progressCallback?.invoke(0.80f, "Writing PLY ($gaussianCount Gaussians)...")
            LogUtil.d(TAG, "inferStreamingSplit: Writing PLY gaussianCount=$gaussianCount roomBounds pending")
            val roomsDir = File(context.filesDir, "sharp_rooms")
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val roomFolder = File(roomsDir, "room_$timestamp")
            roomFolder.mkdirs()
            val plyFile = File(roomFolder, "room.ply")
            val classicPlyFile = File(roomFolder, "room_classic.ply")
            val writeStart = System.currentTimeMillis()
            writePly(plyFile, packedOutput, gaussianCount)
            plyFile.copyTo(classicPlyFile, overwrite = true)
            LogUtil.d(TAG, "inferStreamingSplit: writePly done in ${System.currentTimeMillis() - writeStart}ms ${plyFile.absolutePath} size=${plyFile.length()}")

            var minX = Float.MAX_VALUE
            var maxX = -Float.MAX_VALUE
            var minY = Float.MAX_VALUE
            var maxY = -Float.MAX_VALUE
            var minZ = Float.MAX_VALUE
            var maxZ = -Float.MAX_VALUE
            for (i in 0 until gaussianCount) {
                val off = i * PARAMS_PER_GAUSSIAN
                val x = packedOutput[off]; val y = packedOutput[off + 1]; val z = packedOutput[off + 2]
                if (x < minX) minX = x; if (x > maxX) maxX = x
                if (y < minY) minY = y; if (y > maxY) maxY = y
                if (z < minZ) minZ = z; if (z > maxZ) maxZ = z
            }
            val roomWidth = maxX - minX
            val roomHeight = maxY - minY
            val roomDepth = maxZ - minZ
            LogUtil.d(TAG, "inferStreamingSplit SUCCESS totalMs=${System.currentTimeMillis() - startTime} gaussianCount=$gaussianCount room ${roomWidth}m x ${roomHeight}m x ${roomDepth}m bounds min=($minX,$minY,$minZ) max=($maxX,$maxY,$maxZ)")
            progressCallback?.invoke(1.0f, "Done!")
            return StreamingResult(
                plyFile = plyFile,
                classicPlyFile = classicPlyFile,
                gaussianCount = gaussianCount,
                roomWidth = roomWidth,
                roomHeight = roomHeight,
                roomDepth = roomDepth
            )
        } catch (e: Exception) {
            LogUtil.e(TAG, "Native .pt SPLIT inference failed after ${System.currentTimeMillis() - startTime}ms", e)
            progressCallback?.invoke(0f, "Error: ${e.message}")
            return null
        }
    }

    private fun writePly(file: File, params: FloatArray, gaussianCount: Int) {
        LogUtil.d(TAG, "writePly ENTER file=${file.absolutePath} gaussianCount=$gaussianCount paramsLen=${params.size}")
        val header = buildString {
            append("ply\nformat binary_little_endian 1.0\n")
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

        val zeroSH = ByteArray(45 * 4)
        val lutScale = (LOGIT_LUT_SIZE - 1).toFloat()

        FileOutputStream(file).use { fos ->
            val channel = fos.channel
            channel.write(ByteBuffer.wrap(header.toByteArray()))

            val batchSize = 512
            val buf = ByteBuffer.allocateDirect(BYTES_PER_VERTEX * batchSize).order(ByteOrder.LITTLE_ENDIAN)

            var processed = 0
            while (processed < gaussianCount) {
                val count = minOf(batchSize, gaussianCount - processed)
                buf.clear()

                for (j in 0 until count) {
                    val off = (processed + j) * PARAMS_PER_GAUSSIAN
                    buf.putFloat(params[off + 0])
                    buf.putFloat(-params[off + 1])
                    buf.putFloat(-params[off + 2])
                    buf.putFloat(0f)
                    buf.putFloat(0f)
                    buf.putFloat(0f)

                    val r = params[off + 11].coerceIn(0f, 1f)
                    val g = params[off + 12].coerceIn(0f, 1f)
                    val b = params[off + 13].coerceIn(0f, 1f)
                    buf.putFloat((r - 0.5f) / SH_C0)
                    buf.putFloat((g - 0.5f) / SH_C0)
                    buf.putFloat((b - 0.5f) / SH_C0)

                    buf.put(zeroSH)

                    val op = params[off + 10].coerceIn(0f, 1f)
                    val lutIdx = (op * lutScale).toInt().coerceIn(0, LOGIT_LUT_SIZE - 1)
                    buf.putFloat(LOGIT_LUT[lutIdx])

                    for (s in 3..5) {
                        buf.putFloat(ln(max(params[off + s], 0.001f)))
                    }

                    val qw = params[off + 6]
                    val qx = params[off + 7]
                    val qy = params[off + 8]
                    val qz = params[off + 9]
                    val mag = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
                    val inv = if (mag > 1e-8f) 1f / mag else 1f
                    buf.putFloat(qw * inv)
                    buf.putFloat(qx * inv)
                    buf.putFloat(qy * inv)
                    buf.putFloat(qz * inv)
                }

                buf.flip()
                buf.limit(count * BYTES_PER_VERTEX)
                while (buf.hasRemaining()) channel.write(buf)

                processed += count
            }
        }
        LogUtil.d(TAG, "writePly DONE file=${file.absolutePath} bytes=${file.length()}")
    }

    fun release() {
        module?.destroy()
        module = null
        isInitialized = false
        LogUtil.d(TAG, "NativePtSharp released")
    }
}
