package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.*

/**
 * Optimized ExecuTorch INT8 implementation for SHARP models.
 * Handles 35-patch multi-scale merging and chunked decoding to prevent OOM.
 */
class ExecutorchInt8Sharp private constructor(private val context: Context) {

    companion object {
        private const val TAG = "ExecutorchInt8Sharp"
        // Image + merged spatial sizes (must match Python export)
        private const val IMAGE_SIZE = 1536
        private const val M_1X = 96   // 1x merged size
        private const val M_05X = 48  // 0.5x merged size
        private const val INPUT_SIZE = IMAGE_SIZE
        private const val PATCH_SIZE = 384
        private const val FEATURE_DIM = 1024
        private const val SPATIAL_SIZE = 24
        private const val PARAMS_PER_GAUSSIAN = 14
        private const val SH_C0 = 0.28209479177387814f
        // Match SHARP save_ply: model outputs linear RGB; PLY stores f_dc in sRGB (convert before SH).
        private const val LINEAR_TO_SRGB_THRESHOLD = 0.0031308f
        private const val FLOATS_PER_VERTEX = 62
        private const val BYTES_PER_VERTEX = FLOATS_PER_VERTEX * 4
        private const val PLY_BATCH_SIZE = 512
        private const val GRID_1X = 5
        private const val GRID_05X = 3
        private const val TOTAL_PATCHES = 35
        private const val PADDING_1X = 3
        private const val PADDING_05X = 6

        private const val MODEL_FILENAME = "sharp_int8.pte"
        private val SPLIT_FILENAMES = arrayOf(
            "sharp_split_part1_int8.pte",
            "sharp_split_part2_int8.pte",
            "sharp_split_part3_int8.pte",
            "sharp_split_part4_int8.pte"
        )
        /** Names of .pte files that may be packaged in assets/models/ for testing. */
        private val ASSET_MODEL_FILENAMES = arrayOf(
            "sharp_split_part1_int8.pte",
            "sharp_split_part2_int8.pte",
            "sharp_split_part3_int8.pte",
            "sharp_split_part4a_chunk_512.pte",
            "sharp_split_part4a_chunk_65.pte",
            "sharp_split_part4b.pte"
        )
        private const val ASSET_MODELS_SUBDIR = "models"

        private const val LOGIT_LUT_SIZE = 1024
        private val LOGIT_LUT = FloatArray(LOGIT_LUT_SIZE) { i ->
            val p = (i.toFloat() / (LOGIT_LUT_SIZE - 1)).coerceIn(1e-4f, 1f - 1e-4f)
            ln(p / (1f - p))
        }

        private const val LN_LUT_SIZE = 2048
        private const val LN_LUT_MIN = 0.001f
        private const val LN_LUT_MAX = 5.0f
        private val LN_LUT_SCALE = (LN_LUT_SIZE - 1).toFloat() / (LN_LUT_MAX - LN_LUT_MIN)
        private val LN_LUT = FloatArray(LN_LUT_SIZE) { i -> ln(LN_LUT_MIN + (LN_LUT_MAX - LN_LUT_MIN) * i / (LN_LUT_SIZE - 1)) }

        private fun lnLut(x: Float): Float {
            if (x <= LN_LUT_MIN) return LN_LUT[0]
            if (x >= LN_LUT_MAX) return LN_LUT[LN_LUT_SIZE - 1]
            return LN_LUT[((x - LN_LUT_MIN) * LN_LUT_SCALE).toInt()]
        }

        @Volatile
        private var instance: ExecutorchInt8Sharp? = null

        fun getInstance(context: Context) = instance ?: synchronized(this) {
            instance ?: ExecutorchInt8Sharp(context.applicationContext).also { instance = it }
        }
    }

    private val mutex = Mutex()
    private var isInitialized = false
    private val modelsDir by lazy { context.getExternalFilesDir("models") ?: File(context.filesDir, "models") }
    private val internalModelsDir by lazy { File(context.filesDir, "models").also { it.mkdirs() } }

    private val plyBatch = ByteBuffer.allocateDirect(BYTES_PER_VERTEX * PLY_BATCH_SIZE).apply { order(ByteOrder.LITTLE_ENDIAN) }
    private val zeroSHBuffer = ByteBuffer.allocateDirect(45 * 4).apply { order(ByteOrder.LITTLE_ENDIAN) }
    private val patchPixelBuffer = ByteBuffer.allocateDirect(PATCH_SIZE * PATCH_SIZE * 4).order(ByteOrder.nativeOrder())
    private val patchFloatBuffer = ByteBuffer.allocateDirect(3 * PATCH_SIZE * PATCH_SIZE * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val imagePixelBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 4).order(ByteOrder.nativeOrder())
    private val imageFloatBuffer = ByteBuffer.allocateDirect(3 * INPUT_SIZE * INPUT_SIZE * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()

    data class StreamingResult(val plyFile: File, val classicPlyFile: File, val gaussianCount: Int, val roomWidth: Float, val roomHeight: Float, val roomDepth: Float)

    fun initialize(): Boolean = runBlocking { mutex.withLock { isInitialized = true; true } }

    /** Copy packaged .pte from assets/models/ to filesDir/models so Module.load can use them (for test APKs). */
    private fun ensureModelsFromAssets() {
        for (filename in ASSET_MODEL_FILENAMES) {
            val dest = File(internalModelsDir, filename)
            if (dest.exists() && dest.length() > 0L) continue
            val assetPath = "$ASSET_MODELS_SUBDIR/$filename"
            try {
                context.assets.open(assetPath).use { input: InputStream ->
                    FileOutputStream(dest).use { output ->
                        input.copyTo(output)
                    }
                }
                Log.d(TAG, "Copied $filename from assets to ${dest.absolutePath}")
            } catch (e: Exception) {
                Log.d(TAG, "Asset $assetPath not present or copy failed: ${e.message}")
            }
        }
    }

    private fun findFile(filename: String): File? {
        val internal = File(internalModelsDir, filename).takeIf { it.exists() && it.length() > 0 }
        return internal ?: File(modelsDir, filename).takeIf { it.exists() && it.length() > 0 }
    }

    /** Report progress 0..1 with engaging messages (aligned with Swift/Android overlay text). */
    private fun report(progress: Float, message: String, progressCallback: ((Float, String) -> Unit)?) {
        progressCallback?.invoke(progress.coerceIn(0f, 1f), message)
    }

    suspend fun inferStreaming(bitmap: Bitmap, progressCallback: ((Float, String) -> Unit)? = null): StreamingResult? = withContext(Dispatchers.IO) {
        mutex.withLock {
            if (!isInitialized) return@withContext null
            ensureModelsFromAssets()

            report(0f, "Preparing…", progressCallback)
            // 1. Prepare multi-scale buffers (1x and 0.5x) to match Python export (96x96, 48x48)
            val scaledBmp = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
            val halfSize = IMAGE_SIZE / 2
            val halfBitmap = Bitmap.createScaledBitmap(scaledBmp, halfSize, halfSize, true)

            val mSize1x = M_1X
            val mSize05x = M_05X
            val latent0 = FloatArray(FEATURE_DIM * mSize1x * mSize1x)
            val latent1 = FloatArray(FEATURE_DIM * mSize1x * mSize1x)
            val x0Feat = FloatArray(FEATURE_DIM * mSize1x * mSize1x)
            val x1Feat = FloatArray(FEATURE_DIM * mSize05x * mSize05x)
            val x2Feat = FloatArray(FEATURE_DIM * SPATIAL_SIZE * SPATIAL_SIZE)
            val tempSpatial = FloatArray(FEATURE_DIM * SPATIAL_SIZE * SPATIAL_SIZE)

            val rowOffL0 = intArrayOf(0)
            val rowOffL1 = intArrayOf(0)
            val rowOffX0 = intArrayOf(0)
            val rowOff05x = intArrayOf(0)
            val colOffs1x = buildColumnOffsets(GRID_1X, PADDING_1X)
            val colOffs05x = buildColumnOffsets(GRID_05X, PADDING_05X)

            report(0.02f, "Warming up…", progressCallback)
            // 2. Encoder Phase (Part 1 & 2)
            Log.d(TAG, "Loading INT8 Part1+Part2...")
            val mod1 = Module.load(findFile("sharp_split_part1_int8.pte")!!.absolutePath, Module.LOAD_MODE_MMAP)
            val mod2 = Module.load(findFile("sharp_split_part2_int8.pte")!!.absolutePath, Module.LOAD_MODE_MMAP)
            Log.d(TAG, "Part1+Part2 loaded. Starting 1x patches (${GRID_1X}x${GRID_1X})...")

            val stride = (IMAGE_SIZE - PATCH_SIZE) / 4
            val total1x = GRID_1X * GRID_1X
            var patchIdx = 0
            for (i in 0 until GRID_1X) for (j in 0 until GRID_1X) {
                val patch = Bitmap.createBitmap(scaledBmp, j * stride, i * stride, PATCH_SIZE, PATCH_SIZE)
                val out1 = mod1.forward(EValue.from(Tensor.fromBlob(preprocess(patch, true), longArrayOf(1, 3, 384, 384))))
                val tokens = out1[0].toTensor().dataAsFloatArray

                reshapeToSpatial(out1[1].toTensor().dataAsFloatArray, tempSpatial)
                mergeCrop(latent0, mSize1x, tempSpatial, j, i, GRID_1X, PADDING_1X, rowOffL0, colOffs1x)

                reshapeToSpatial(tokens, tempSpatial)
                mergeCrop(latent1, mSize1x, tempSpatial, j, i, GRID_1X, PADDING_1X, rowOffL1, colOffs1x)

                val feat = mod2.forward(EValue.from(Tensor.fromBlob(tokens, longArrayOf(1, 577, 1024))))[0].toTensor().dataAsFloatArray
                mergeCrop(x0Feat, mSize1x, feat, j, i, GRID_1X, PADDING_1X, rowOffX0, colOffs1x)
                patch.recycle()
                patchIdx++
                report(0.05f + 0.30f * (patchIdx.toFloat() / total1x), "Building your room… step $patchIdx of $total1x", progressCallback)
            }

            Log.d(TAG, "1x patches done. Starting 0.5x patches (${GRID_05X}x${GRID_05X})...")
            val total05x = GRID_05X * GRID_05X
            var patch05Idx = 0
            val stride05x = (halfSize - PATCH_SIZE) / 2
            for (i in 0 until GRID_05X) {
                for (j in 0 until GRID_05X) {
                    val patch = Bitmap.createBitmap(halfBitmap, j * stride05x, i * stride05x, PATCH_SIZE, PATCH_SIZE)
                    val out05 = mod1.forward(EValue.from(Tensor.fromBlob(preprocess(patch, true), longArrayOf(1, 3, PATCH_SIZE.toLong(), PATCH_SIZE.toLong()))))
                    if (out05.isEmpty()) {
                        Log.e(TAG, "Part1 returned no outputs at 0.5x i=$i j=$j")
                        patch.recycle(); halfBitmap.recycle(); mod1.destroy(); mod2.destroy()
                        return@withContext null
                    }
                    val tokens = out05[0].toTensor().dataAsFloatArray
                    val feat05 = mod2.forward(EValue.from(Tensor.fromBlob(tokens, longArrayOf(1, 577, 1024))))[0].toTensor().dataAsFloatArray
                    mergeCrop(x1Feat, mSize05x, feat05, j, i, GRID_05X, PADDING_05X, rowOff05x, colOffs05x)
                    patch.recycle()
                    patch05Idx++
                    report(0.36f + 0.06f * (patch05Idx.toFloat() / total05x), "Building your room…", progressCallback)
                }
            }
            halfBitmap.recycle()
            report(0.43f, "Building your room…", progressCallback)
            Log.d(TAG, "0.5x patches done. Running 0.25x patch (35th) for x2Feat...")
            val quarterBmp = Bitmap.createScaledBitmap(scaledBmp, 384, 384, true)
            patchFloatBuffer.rewind()
            val qOut = mod1.forward(EValue.from(Tensor.fromBlob(preprocess(quarterBmp, true), longArrayOf(1, 3, 384, 384))))
            if (qOut.isEmpty()) {
                Log.e(TAG, "Part1 returned no outputs for 0.25x patch")
                quarterBmp.recycle(); mod1.destroy(); mod2.destroy()
                return@withContext null
            }
            val qTokens = qOut[0].toTensor()
            val qFeat = mod2.forward(EValue.from(qTokens))[0].toTensor().dataAsFloatArray
            reshapeToSpatial(qFeat, x2Feat)
            quarterBmp.recycle()
            Log.d(TAG, "0.25x patch done. Destroying Part1+Part2...")
            mod1.destroy(); mod2.destroy(); System.gc()

            report(0.44f, "Understanding the full picture…", progressCallback)
            // 3. Image Encoder (Part 3)
            Log.d(TAG, "Loading Part3 (image encoder)...")
            val mod3 = Module.load(findFile("sharp_split_part3_int8.pte")!!.absolutePath)
            imageFloatBuffer.rewind()
            val fullImgTensor = Tensor.fromBlob(preprocess(scaledBmp, false), longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong()))
            val imgTokens = mod3.forward(EValue.from(fullImgTensor))[0].toTensor().dataAsFloatArray
            Log.d(TAG, "Part3 done. imgTokens size=${imgTokens.size}")
            mod3.destroy(); System.gc()
            report(0.46f, "Understanding the full picture…", progressCallback)

            report(0.48f, "Adding depth and shape…", progressCallback)
            // 4. Chunked Part 4a (tokens 512 + 65); sliceArray for buffer safety with Tensor.fromBlob
            Log.d(TAG, "Starting chunked Part4 decoder...")
            val out512 = runDecoderChunk("sharp_split_part4a_chunk_512.pte", imgTokens.sliceArray(0 until 512 * 1024), 512)
            report(0.49f, "Adding depth and shape…", progressCallback)
            val out65 = runDecoderChunk("sharp_split_part4a_chunk_65.pte", imgTokens.sliceArray(512 * 1024 until 577 * 1024), 65)
            val combinedTokens = out512 + out65
            Log.d(TAG, "Part4a chunks done. combinedTokens size=${combinedTokens.size}. Loading Part4b...")

            report(0.50f, "Adding the finishing touches…", progressCallback)
            val part4bThreads = Runtime.getRuntime().availableProcessors().coerceIn(2, 8)
            val mod4b = Module.load(
                findFile("sharp_split_part4b.pte")!!.absolutePath,
                Module.LOAD_MODE_MMAP,
                part4bThreads
            )
            Log.d(TAG, "Part4b forward: 7 inputs tokens, img, latent0, latent1, x0Feat, x1Feat, x2Feat")
            val part4bInputs = listOf(
                EValue.from(Tensor.fromBlob(combinedTokens, longArrayOf(1, 577, 1024))),
                EValue.from(fullImgTensor),
                EValue.from(Tensor.fromBlob(latent0, longArrayOf(1, 1024, 96, 96))),
                EValue.from(Tensor.fromBlob(latent1, longArrayOf(1, 1024, 96, 96))),
                EValue.from(Tensor.fromBlob(x0Feat, longArrayOf(1, 1024, 96, 96))),
                EValue.from(Tensor.fromBlob(x1Feat, longArrayOf(1, 1024, 48, 48))),
                EValue.from(Tensor.fromBlob(x2Feat, longArrayOf(1, 1024, 24, 24)))
            )
            val part4bEstimatedSeconds = 80f
            val part4bProgressStart = 0.55f
            val part4bProgressSpan = 0.35f
            val finalParams = coroutineScope {
                val deferred = async(Dispatchers.IO) {
                    mod4b.forward(*part4bInputs.toTypedArray())[0].toTensor()
                }
                val startMs = System.currentTimeMillis()
                while (!deferred.isCompleted) {
                    delay(2000)
                    if (deferred.isCompleted) break
                    val elapsedSec = (System.currentTimeMillis() - startMs) / 1000f
                    val p = part4bProgressStart + (elapsedSec / part4bEstimatedSeconds).coerceIn(0f, 1f) * part4bProgressSpan
                    report(p, "Adding the finishing touches… This may take a minute.", progressCallback)
                }
                deferred.await()
            }
            report(0.92f, "Saving your 3D room…", progressCallback)
            Log.d(TAG, "Part4b done. Writing PLY...")
            val result = writePly(finalParams.dataAsFloatArray, progressCallback)
            mod4b.destroy(); scaledBmp.recycle()
            report(1f, "Your room is ready!", progressCallback)
            Log.d(TAG, "Inference complete. Gaussians=${result.gaussianCount} room=${result.roomWidth}x${result.roomHeight}x${result.roomDepth}")
            return@withContext result
        }
    }

    private fun runDecoderChunk(name: String, data: FloatArray, count: Int): FloatArray {
        Log.d(TAG, "runDecoderChunk: $name count=$count dataLen=${data.size}")
        val mod = Module.load(findFile(name)!!.absolutePath)
        patchFloatBuffer.rewind()
        val inputTensor = Tensor.fromBlob(data, longArrayOf(1, count.toLong(), 1024))
        val output = mod.forward(EValue.from(inputTensor))
        if (output.isEmpty()) {
            Log.e(TAG, "runDecoderChunk: $name returned no outputs")
            mod.destroy()
            return FloatArray(0)
        }
        val out = output[0].toTensor().dataAsFloatArray
        Log.d(TAG, "runDecoderChunk: $name outputLen=${out.size}")
        mod.destroy()
        System.gc()
        return out
    }

    private fun preprocess(bmp: Bitmap, isPatch: Boolean): FloatBuffer {
        val buf = if (isPatch) patchFloatBuffer else imageFloatBuffer
        val pix = if (isPatch) patchPixelBuffer else imagePixelBuffer
        val sz = if (isPatch) PATCH_SIZE else INPUT_SIZE
        pix.clear(); bmp.copyPixelsToBuffer(pix); pix.rewind(); buf.clear()
        val total = sz * sz
        for (i in 0 until total) {
            val argb = pix.getInt()
            buf.put(i, (argb shr 16 and 0xFF) / 255f)
            buf.put(total + i, (argb shr 8 and 0xFF) / 255f)
            buf.put(2 * total + i, (argb and 0xFF) / 255f)
        }
        return buf.rewind() as FloatBuffer
    }

    private fun mergeCrop(out: FloatArray, outW: Int, patch: FloatArray, gI: Int, gJ: Int, gS: Int, pad: Int, offY: IntArray, offX: IntArray) {
        val cT = if (gJ != 0) pad else 0; val cB = if (gJ != gS - 1) pad else 0
        val cL = if (gI != 0) pad else 0; val cR = if (gI != gS - 1) pad else 0
        val cH = SPATIAL_SIZE - cT - cB; val cW = SPATIAL_SIZE - cL - cR
        val outHW = outW * outW
        for (c in 0 until FEATURE_DIM) {
            val sB = c * 576; val dB = c * outHW
            for (dy in 0 until cH) for (dx in 0 until cW) {
                out[dB + (offY[0] + dy) * outW + (offX[gI] + dx)] = patch[sB + (cT + dy) * 24 + (cL + dx)]
            }
        }
        if (gI == gS - 1) offY[0] += cH
    }

    /**
     * Maps token sequence to spatial [C, H, W]. Out buffer is 24*24*1024 = 589824.
     * If tokens.size == 577*1024 (CLS + 576 spatial), skip CLS with +1. If tokens.size == 576*1024 (no CLS), use 0-based index.
     */
    private fun reshapeToSpatial(tokens: FloatArray, out: FloatArray, H: Int = 24, W: Int = 24) {
        val C = 1024
        val spatialCount = H * W
        val hasCls = (tokens.size == (spatialCount + 1) * C) // 577*1024 -> skip index 0
        val tokenOffset = if (hasCls) 1 else 0
        for (h in 0 until H) {
            for (w in 0 until W) {
                val tokenIdx = (h * W + w) + tokenOffset
                val outBase = (h * W + w)
                for (c in 0 until C) {
                    out[c * (H * W) + outBase] = tokens[tokenIdx * C + c]
                }
            }
        }
    }

    private fun buildColumnOffsets(gS: Int, pad: Int) = IntArray(gS).apply {
        var x = 0; for (i in 0 until gS) { this[i] = x; x += 24 - (if (i != 0) pad else 0) - (if (i != gS - 1) pad else 0) }
    }

    /** Linear RGB [0,1] -> sRGB [0,1]. Matches SHARP save_ply (Metal spec 7.7.7) so PLY f_dc matches viewer expectation. */
    private fun linearToSrgb(linear: Float): Float {
        val v = linear.coerceIn(0f, 1f)
        return if (v <= LINEAR_TO_SRGB_THRESHOLD) v * 12.92f
        else (1.055f * v.toDouble().pow(1.0 / 2.4).toFloat() - 0.055f).coerceIn(0f, 1f)
    }

    private fun writePly(params: FloatArray, progressCallback: ((Float, String) -> Unit)?): StreamingResult {
        val count = params.size / PARAMS_PER_GAUSSIAN
        val roomFolder = File(File(context.filesDir, "sharp_rooms"), "room_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}").apply { mkdirs() }
        val plyFile = File(roomFolder, "room.ply")
        val progressReportEvery = (count / 8).coerceAtLeast(1)

        var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
        var minZ = Float.MAX_VALUE; var maxZ = -Float.MAX_VALUE

        FileOutputStream(plyFile).use { fos ->
            val channel = fos.channel
            val header = "ply\nformat binary_little_endian 1.0\nelement vertex $count\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\n" +
                    (0 until 3).joinToString("") { "property float f_dc_$it\n" } + (0 until 45).joinToString("") { "property float f_rest_$it\n" } +
                    "property float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nend_header\n"
            channel.write(ByteBuffer.allocateDirect(header.length).apply { put(header.toByteArray()); flip() })

            for (i in 0 until count) {
                if (i > 0 && i % progressReportEvery == 0) {
                    report(0.92f + 0.08f * (i.toFloat() / count), "Saving your 3D room…", progressCallback)
                }
                val off = i * PARAMS_PER_GAUSSIAN
                val x = params[off]; val y = -params[off + 1]; val z = -params[off + 2]
                minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y); minZ = min(minZ, z); maxZ = max(maxZ, z)

                plyBatch.clear()
                plyBatch.putFloat(x).putFloat(y).putFloat(z).putFloat(0f).putFloat(0f).putFloat(0f)
                // Model outputs color as BGR at 11,12,13; PLY f_dc is RGB — swap so brown isn’t blue
                val b = linearToSrgb(params[off + 11])
                val g = linearToSrgb(params[off + 12])
                val r = linearToSrgb(params[off + 13])
                plyBatch.putFloat((r - 0.5f) / SH_C0).putFloat((g - 0.5f) / SH_C0).putFloat((b - 0.5f) / SH_C0)
                zeroSHBuffer.rewind(); plyBatch.put(zeroSHBuffer)
                plyBatch.putFloat(LOGIT_LUT[(params[off + 3] * 1023).toInt().coerceIn(0, 1023)])
                plyBatch.putFloat(lnLut(max(params[off + 4] * 1.3f, 0.001f))).putFloat(lnLut(max(params[off + 5] * 1.3f, 0.001f))).putFloat(lnLut(max(params[off + 6] * 1.3f, 0.001f)))

                val rw = params[off + 7]; val rx = params[off + 8]; val ry = params[off + 9]; val rz = params[off + 10]
                val m = sqrt(rw * rw + rx * rx + ry * ry + rz * rz).let { if (it > 1e-8f) 1f / it else 1f }
                plyBatch.putFloat(rw * m).putFloat(rx * m).putFloat(ry * m).putFloat(rz * m)

                plyBatch.flip(); while (plyBatch.hasRemaining()) channel.write(plyBatch)
            }
        }
        return StreamingResult(plyFile, plyFile, count, maxX - minX, maxY - minY, maxZ - minZ)
    }
}

