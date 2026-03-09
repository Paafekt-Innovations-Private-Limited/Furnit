package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import com.furnit.android.utils.LogUtil
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
 *
 * Input: center-crop to square (min(w,h)) then resize to 1536x1536. We do not pad gray to make a bigger square
 * (e.g. 2x4 -> 4x4 with gray on sides): that was tried (114-gray letterbox) and caused jagged output—ViT sees
 * the content/padding edge as OOD; INT8 amplifies it. Raw PLY coords; no aspect scale.
 *
 * Export note: if jagged output persists, consider FP16 for position/scale decoder heads
 * (see Ultralytics deployment practices; INT8 on scales/rotations can cause severe artifacts).
 */
class ExecutorchInt8Sharp private constructor(private val context: Context) {

    companion object {
        private const val TAG = "ExecutorchInt8Sharp"
        /** Use Lanczos3 for center-crop resize when true; bilinear when false. Set false if Lanczos is too slow or causes issues. */
        @JvmField
        var USE_LANCZOS_RESIZE: Boolean = true
        /** When true, stretch full image to 1536x1536 (like Swift; no crop). When false, center-crop to square then resize. Set false if stretch causes jagged output on INT8. */
        @JvmField
        var USE_STRETCH_TO_SQUARE: Boolean = true
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
            "sharp_split_part4b.pte",
            "sharp_split_part4b_tile_full.pte",
            "sharp_split_part4b_tile_00.pte", "sharp_split_part4b_tile_01.pte", "sharp_split_part4b_tile_02.pte", "sharp_split_part4b_tile_03.pte",
            "sharp_split_part4b_tile_04.pte", "sharp_split_part4b_tile_05.pte", "sharp_split_part4b_tile_06.pte", "sharp_split_part4b_tile_07.pte",
            "sharp_split_part4b_tile_08.pte", "sharp_split_part4b_tile_09.pte", "sharp_split_part4b_tile_10.pte", "sharp_split_part4b_tile_11.pte",
            "sharp_split_part4b_tile_12.pte", "sharp_split_part4b_tile_13.pte", "sharp_split_part4b_tile_14.pte", "sharp_split_part4b_tile_15.pte"
        )
        private const val PART4B_TILED_GRID = 4
        private const val PART4B_TILED_NUM = PART4B_TILED_GRID * PART4B_TILED_GRID // 16
        private const val PART4B_GAUSSIANS_PER_TILE = 73728
        /** When true, native C++ may use 2 modules in parallel (ignored in C++ for stability); read from prefs in inferStreaming. */
        @JvmField
        var PARALLEL_TILES: Boolean = false
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

        /** Whether the native 16-tile Part4b library loaded successfully. When false, Kotlin/single path is used. */
        @JvmField
        var NATIVE_TILES_AVAILABLE: Boolean = false

        /** Whether the native full INT8 pipeline library loaded (Part1–4b in C++, single Part4b). When true and pref on, inferStreaming tries C++ first. */
        @JvmField
        var NATIVE_FULL_AVAILABLE: Boolean = false

        init {
            try {
                System.loadLibrary("sharp_executorch_tiles")
                NATIVE_TILES_AVAILABLE = true
            } catch (e: UnsatisfiedLinkError) {
                LogUtil.d(TAG, "Native tile library not available: ${e.message}")
            }
            if (NATIVE_TILES_AVAILABLE) {
                LogUtil.d(TAG, "Native Part4b tile library loaded (16-tile C++ path when Part4b tiled is enabled)")
            }
            try {
                System.loadLibrary("sharp_executorch_full")
                NATIVE_FULL_AVAILABLE = true
            } catch (e: UnsatisfiedLinkError) {
                LogUtil.d(TAG, "Native full pipeline library not available: ${e.message}")
            }
            if (NATIVE_FULL_AVAILABLE) {
                LogUtil.d(TAG, "Native full INT8 pipeline loaded (C++ ExecuTorch INT8 option in Settings)")
            }
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

    data class StreamingResult(
        val plyFile: File,
        val classicPlyFile: File,
        val gaussianCount: Int,
        val roomWidth: Float,
        val roomHeight: Float,
        val roomDepth: Float,
        val roomCenterX: Float,
        val roomCenterY: Float,
        val roomCenterZ: Float
    )

    fun initialize(): Boolean = runBlocking {
        mutex.withLock {
            isInitialized = true
            if (NATIVE_FULL_AVAILABLE) {
                val prefs = context.getSharedPreferences("furnit_prefs", Context.MODE_PRIVATE)
                if (prefs.getBoolean("executorch_int8_use_cpp_full_pipeline", false)) {
                    val part1File = findFile("sharp_split_part1_int8.pte")
                    val cppModelDir = part1File?.parent
                    if (cppModelDir != null) {
                        val preloaded = preloadCppModules(cppModelDir)
                        LogUtil.d(TAG, "[C++ FULL] Preload Part1+Part2 cache: ${if (preloaded) "OK" else "failed"} (dir=$cppModelDir)")
                    }
                }
            }
            true
        }
    }

    /** Copy packaged .pte from assets/models/ to filesDir/models so Module.load can use them. Loads all 16 Part4b tiles from assets when present (ODR-style: packaged in APK, extracted on first use). */
    private fun ensureModelsFromAssets() {
        var tilesCopied = 0
        var tilesMissing = 0
        for (filename in ASSET_MODEL_FILENAMES) {
            val dest = File(internalModelsDir, filename)
            if (dest.exists() && dest.length() > 0L) {
                if (filename.startsWith("sharp_split_part4b_tile_")) tilesCopied++
                continue
            }
            val assetPath = "$ASSET_MODELS_SUBDIR/$filename"
            val isTile = filename.startsWith("sharp_split_part4b_tile_")
            try {
                context.assets.open(assetPath).use { input: InputStream ->
                    FileOutputStream(dest).use { output ->
                        input.copyTo(output)
                    }
                }
                if (isTile) tilesCopied++
                LogUtil.d(TAG, "Copied $filename from assets to ${dest.absolutePath}")
            } catch (e: Exception) {
                if (isTile) tilesMissing++
                else LogUtil.d(TAG, "Asset $assetPath not present or copy failed: ${e.message}")
            }
        }
        if (tilesMissing > 0) {
            LogUtil.d(TAG, "Part4b tile models: ${tilesCopied}/17 from assets (add tile .pte to executorch_int8_models or executorch_models and rebuild for tiled path)")
        } else if (tilesCopied == 17) {
            LogUtil.d(TAG, "Part4b tile models ready from assets (16 tiles + tile_full)")
        }
    }

    /**
     * Part4b thread count: fixed 2 so behavior matches 8 GB phone and does not OOM on any device
     * (12 GB and others). More threads can be re-enabled later for simulator/testing.
     */
    private fun part4bThreadCount(): Int = 2

    private fun findFile(filename: String): File? {
        val internal = File(internalModelsDir, filename).takeIf { it.exists() && it.length() > 0 }
        return internal ?: File(modelsDir, filename).takeIf { it.exists() && it.length() > 0 }
    }

    /** Crop tile (row,col) from NCHW flat array. shape=[1,C,H,W], output=[C*(H/n)*(W/n)]. */
    private fun cropTileNCHW(data: FloatArray, channels: Int, fullH: Int, fullW: Int, tileRow: Int, tileCol: Int): FloatArray {
        val tileH = fullH / PART4B_TILED_GRID
        val tileW = fullW / PART4B_TILED_GRID
        val out = FloatArray(channels * tileH * tileW)
        for (c in 0 until channels) for (y in 0 until tileH) for (x in 0 until tileW) {
            out[c * tileH * tileW + y * tileW + x] =
                data[c * fullH * fullW + (tileRow * tileH + y) * fullW + (tileCol * tileW + x)]
        }
        return out
    }

    /** NDC correction for a tile's packed [x,y,z,...] Gaussians so stitched tiles align. swapNdcXY: use tileRow for X and tileCol for Y (if export uses transposed tile layout). */
    private fun correctNDC(packedTile: FloatArray, tileRow: Int, tileCol: Int, swapNdcXY: Boolean = false) {
        val grid = PART4B_TILED_GRID
        val numGaussians = packedTile.size / 14
        val ndcOffsetX = (2f * tileCol + 1f - grid) / grid
        val ndcOffsetY = (2f * tileRow + 1f - grid) / grid
        val invGrid = 1f / grid
        val offX = if (swapNdcXY) ndcOffsetY else ndcOffsetX
        val offY = if (swapNdcXY) ndcOffsetX else ndcOffsetY
        for (g in 0 until numGaussians) {
            val base = g * 14
            val posZ = packedTile[base + 2]
            packedTile[base + 0] = packedTile[base + 0] * invGrid + posZ * offX
            packedTile[base + 1] = packedTile[base + 1] * invGrid + posZ * offY
            packedTile[base + 4] *= invGrid
            packedTile[base + 5] *= invGrid
            packedTile[base + 6] *= invGrid
        }
    }

    /**
     * Run full INT8 pipeline in C++ (Part1, Part2, Part3, Part4a, Part4b single). Returns packed [N*14] Gaussian floats,
     * or null if not implemented or on error (caller falls back to Kotlin pipeline).
     */
    private external fun runFullPipelineInt8Native(modelDirPath: String, imageNCHW: FloatArray): FloatArray?

    /** Warm-start: preload Part1+Part2 into native singleton cache (eliminates ~400ms mmap on first inference). */
    private external fun preloadCppModules(modelDirPath: String): Boolean

    /** Release native singleton Part1+Part2 cache and aligned workspace buffers. */
    private external fun releaseCppModules()

    /**
     * Run the 16-tile Part4b pipeline in C++ (native heap). Returns packed [N*14] Gaussian floats,
     * or null on failure (caller falls back to Kotlin tile loop or single Part4b).
     */
    private external fun runTiledPart4bNative(
        modelPath: String,
        imageNCHW: FloatArray,
        latent0: FloatArray,
        latent1: FloatArray,
        x0Feat: FloatArray,
        x1Feat: FloatArray,
        x2Feat: FloatArray,
        xLowres: FloatArray,
        numThreads: Int,
        parallelTiles: Boolean,
        swapNdcXY: Boolean
    ): FloatArray?

    /** Report progress 0..1 with engaging messages (aligned with Swift/Android overlay text). */
    private fun report(progress: Float, message: String, progressCallback: ((Float, String) -> Unit)?) {
        progressCallback?.invoke(progress.coerceIn(0f, 1f), message)
    }

    /** Lanczos3 kernel: sinc(x)*sinc(x/3) for |x|<3, else 0. */
    private fun lanczos3(x: Double): Double {
        if (x <= -3.0 || x >= 3.0) return 0.0
        if (x == 0.0) return 1.0
        val px = PI * x
        return (sin(px) / px) * (sin(px / 3.0) / (px / 3.0))
    }

    /**
     * Resize bitmap using Lanczos3 (6x6 kernel). Slower than bilinear; use for higher-quality resize.
     * Output clamped to 0..255 to match bilinear range for INT8 input (per Ultralytics).
     */
    private fun resizeWithLanczos3(source: Bitmap, targetW: Int, targetH: Int): Bitmap {
        val sw = source.width
        val sh = source.height
        val srcPixels = IntArray(sw * sh)
        source.getPixels(srcPixels, 0, sw, 0, 0, sw, sh)
        val dstPixels = IntArray(targetW * targetH)
        val scaleX = sw.toDouble() / targetW
        val scaleY = sh.toDouble() / targetH
        for (oy in 0 until targetH) {
            for (ox in 0 until targetW) {
                val sx = (ox + 0.5) * scaleX - 0.5
                val sy = (oy + 0.5) * scaleY - 0.5
                var r = 0.0
                var g = 0.0
                var b = 0.0
                var wSum = 0.0
                val ix0 = max(0, floor(sx).toInt() - 2)
                val ix1 = min(sw - 1, ceil(sx).toInt() + 3)
                val iy0 = max(0, floor(sy).toInt() - 2)
                val iy1 = min(sh - 1, ceil(sy).toInt() + 3)
                for (iy in iy0..iy1) {
                    val wy = lanczos3(sy - iy)
                    if (wy == 0.0) continue
                    for (ix in ix0..ix1) {
                        val wx = lanczos3(sx - ix)
                        if (wx == 0.0) continue
                        val w = wx * wy
                        val pixel = srcPixels[iy * sw + ix]
                        r += (pixel shr 16 and 0xFF) * w
                        g += (pixel shr 8 and 0xFF) * w
                        b += (pixel and 0xFF) * w
                        wSum += w
                    }
                }
                val n = if (wSum > 0.0) wSum else 1.0
                val rr = (r / n).toInt().coerceIn(0, 255)
                val gg = (g / n).toInt().coerceIn(0, 255)
                val bb = (b / n).toInt().coerceIn(0, 255)
                dstPixels[oy * targetW + ox] = 0xFF000000.toInt() or (rr shl 16) or (gg shl 8) or bb
            }
        }
        val result = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        result.setPixels(dstPixels, 0, targetW, 0, 0, targetW, targetH)
        return result
    }

    /**
     * Stretch full image to targetSize x targetSize (no crop). Like Swift; aspect distortion, continuous signal.
     * May cause artifacts on INT8; set USE_STRETCH_TO_SQUARE=false to fall back to center-crop.
     */
    private fun getStretchToSquareBitmap(bitmap: Bitmap, targetSize: Int): Bitmap {
        val result = Bitmap.createScaledBitmap(bitmap, targetSize, targetSize, true)
        LogUtil.d(TAG, "[STRETCH] input=${bitmap.width}x${bitmap.height} -> ${targetSize}x${targetSize} (full image, no crop)")
        return result
    }

    /**
     * Center-crop to square (side = min(w,h)) then resize to targetSize. Matches ViT training distribution;
     * avoids letterbox/gray padding which causes jagged output (Ultralytics).
     * When USE_LANCZOS_RESIZE is true, resize uses Lanczos3; else bilinear. Set USE_LANCZOS_RESIZE=false if slow or problematic.
     */
    private fun getCenterCropBitmap(bitmap: Bitmap, targetSize: Int): Bitmap {
        val side = min(bitmap.width, bitmap.height).coerceAtLeast(1)
        val left = (bitmap.width - side) / 2
        val top = (bitmap.height - side) / 2
        val cropped = Bitmap.createBitmap(bitmap, left, top, side, side)
        val result = if (USE_LANCZOS_RESIZE && side != targetSize) {
            resizeWithLanczos3(cropped, targetSize, targetSize)
        } else {
            Bitmap.createScaledBitmap(cropped, targetSize, targetSize, true)
        }
        if (cropped != bitmap) cropped.recycle()
        LogUtil.d(TAG, "[CENTER_CROP] input=${bitmap.width}x${bitmap.height} -> crop ${side}x${side} -> ${targetSize}x${targetSize} (${if (USE_LANCZOS_RESIZE) "Lanczos3" else "bilinear"})")
        return result
    }

    suspend fun inferStreaming(bitmap: Bitmap, progressCallback: ((Float, String) -> Unit)? = null): StreamingResult? = withContext(Dispatchers.IO) {
        mutex.withLock {
            if (!isInitialized) return@withContext null
            ensureModelsFromAssets()

            // Require Part1–3 and Part4a chunks; Part4b can be single .pte or chunked/tiled when those paths exist
            val required = listOf(
                "sharp_split_part1_int8.pte",
                "sharp_split_part2_int8.pte",
                "sharp_split_part3_int8.pte",
                "sharp_split_part4a_chunk_512.pte",
                "sharp_split_part4a_chunk_65.pte"
            )
            val part4bFile = findFile("sharp_split_part4b.pte")
            val part4bTileFullFile = findFile("sharp_split_part4b_tile_full.pte")
            val part4bTileFiles = (0 until PART4B_TILED_NUM).map { findFile("sharp_split_part4b_tile_%02d.pte".format(it)) }
            val hasAll16TileFiles = part4bTileFiles.all { it != null }
            val prefs = context.getSharedPreferences("furnit_prefs", Context.MODE_PRIVATE)
            val useCppFullPipeline = prefs.getBoolean("executorch_int8_use_cpp_full_pipeline", false)
            val usePart4bTiledPref = prefs.getBoolean("executorch_int8_use_part4b_tiled", false)
            val skipTiledAfterOom = prefs.getBoolean("executorch_int8_skip_tiled_after_oom", false)
            val preferSinglePart4b = prefs.getBoolean("executorch_int8_prefer_single_part4b", false)
            PARALLEL_TILES = prefs.getBoolean("executorch_int8_parallel_tiles", false)
            val swapNdcXY = prefs.getBoolean("executorch_int8_swap_tile_ndc_xy", false)
            val useChunkedPart4bTiled16Files = usePart4bTiledPref && !skipTiledAfterOom && (hasAll16TileFiles || part4bTileFullFile != null) && !preferSinglePart4b
            val needPart4b = !useChunkedPart4bTiled16Files
            val missing = required.filter { findFile(it) == null }
            if (missing.isNotEmpty() || (needPart4b && part4bFile == null)) {
                val missingPart4b = if (needPart4b && part4bFile == null)
                    listOf("sharp_split_part4b.pte")
                else emptyList()
                LogUtil.e(TAG, "ExecuTorch INT8 models missing: ${missing + missingPart4b}. Push to ${modelsDir.absolutePath} and retry.")
                return@withContext null
            }
            if (useChunkedPart4bTiled16Files && part4bFile == null) {
                LogUtil.e(TAG, "Part4b tiled requested but sharp_split_part4b.pte missing (needed for fallback). Push part4b.pte to ${modelsDir.absolutePath}")
                return@withContext null
            }
            when {
                useChunkedPart4bTiled16Files -> LogUtil.d(TAG, "Part4b: using ${if (hasAll16TileFiles) "16 fused tile files" else "tile_full.pte 16x"} (${if (NATIVE_TILES_AVAILABLE) "native C++" else "Kotlin fallback"})")
                preferSinglePart4b -> LogUtil.d(TAG, "Part4b: using single (Stable mode ON in Settings)")
                usePart4bTiledPref && !skipTiledAfterOom -> LogUtil.d(TAG, "Part4b: tiled requested but tile models missing on device; using single Part4b")
                else -> LogUtil.d(TAG, "Part4b: using single ${part4bFile!!.name} (FP32)")
            }

            val originalWidth = bitmap.width
            val originalHeight = bitmap.height
            val isPortrait = originalHeight > originalWidth
            val pipelineStartMs = System.currentTimeMillis()
            LogUtil.d(TAG, "[ASPECT] input=${originalWidth}x${originalHeight} | isPortrait=$isPortrait | ${if (USE_STRETCH_TO_SQUARE) "stretch-to-square" else "center-crop"} + raw coords (no aspect scale in PLY)")

            report(0f, "Preparing…", progressCallback)
            // 1. Stretch full image to 1536 (like Swift) or center-crop then resize; raw PLY coords
            val scaledBmp = if (USE_STRETCH_TO_SQUARE) getStretchToSquareBitmap(bitmap, IMAGE_SIZE) else getCenterCropBitmap(bitmap, IMAGE_SIZE)

            // Optional: run full pipeline in C++ (Part1–4b single, no tiles). Use same model dir as Kotlin (internal or external).
            if (useCppFullPipeline && NATIVE_FULL_AVAILABLE) {
                val part1File = findFile("sharp_split_part1_int8.pte")
                val cppModelDir = part1File?.parent ?: internalModelsDir.absolutePath
                LogUtil.d(TAG, "[C++ FULL] Starting C++ full pipeline, modelDir=$cppModelDir")
                preprocess(scaledBmp, false)
                imageFloatBuffer.rewind()
                val imageNCHW = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE)
                imageFloatBuffer.get(imageNCHW)
                LogUtil.d(TAG, "[C++ FULL] Image preprocessed, calling native (${imageNCHW.size} floats)")
                try {
                    val cppStartMs = System.currentTimeMillis()
                    val cppResult = runFullPipelineInt8Native(cppModelDir, imageNCHW)
                    val cppElapsedMs = System.currentTimeMillis() - cppStartMs
                    if (cppResult != null && cppResult.isNotEmpty()) {
                        LogUtil.d(TAG, "[C++ FULL] Native returned ${cppResult.size / 14} Gaussians in ${cppElapsedMs}ms")
                        report(0.92f, "Saving your 3D room…", progressCallback)
                        val result = writePly(cppResult, progressCallback, isPortrait)
                        scaledBmp.recycle()
                        report(1f, "Your room is ready!", progressCallback)
                        LogUtil.d(TAG, "[TIMING] C++ full pipeline: Gaussians=${result.gaussianCount}, total=${cppElapsedMs}ms")
                        return@withContext result
                    } else {
                        LogUtil.d(TAG, "[C++ FULL] Native returned null/empty after ${cppElapsedMs}ms, falling back to Kotlin")
                    }
                } catch (e: Throwable) {
                    LogUtil.e(TAG, "[C++ FULL] C++ pipeline failed, falling back to Kotlin: ${e.message}", e)
                }
            }

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
            val tPart12Load = System.currentTimeMillis()
            // 2. Encoder Phase (Part 1 & 2)
            LogUtil.d(TAG, "Loading INT8 Part1+Part2...")
            val mod1 = Module.load(findFile("sharp_split_part1_int8.pte")!!.absolutePath, Module.LOAD_MODE_MMAP)
            val mod2 = Module.load(findFile("sharp_split_part2_int8.pte")!!.absolutePath, Module.LOAD_MODE_MMAP)
            LogUtil.d(TAG, "Part1+Part2 loaded in ${System.currentTimeMillis() - tPart12Load}ms. Starting 1x patches (${GRID_1X}x${GRID_1X})...")
            val t1xStart = System.currentTimeMillis()

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

            LogUtil.d(TAG, "1x patches done in ${System.currentTimeMillis() - t1xStart}ms. Starting 0.5x patches (${GRID_05X}x${GRID_05X})...")
            val t05xStart = System.currentTimeMillis()
            val total05x = GRID_05X * GRID_05X
            var patch05Idx = 0
            val stride05x = (halfSize - PATCH_SIZE) / 2
            for (i in 0 until GRID_05X) {
                for (j in 0 until GRID_05X) {
                    val patch = Bitmap.createBitmap(halfBitmap, j * stride05x, i * stride05x, PATCH_SIZE, PATCH_SIZE)
                    val out05 = mod1.forward(EValue.from(Tensor.fromBlob(preprocess(patch, true), longArrayOf(1, 3, PATCH_SIZE.toLong(), PATCH_SIZE.toLong()))))
                    if (out05.isEmpty()) {
                        LogUtil.e(TAG, "Part1 returned no outputs at 0.5x i=$i j=$j")
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
            LogUtil.d(TAG, "0.5x patches done in ${System.currentTimeMillis() - t05xStart}ms. Running 0.25x patch (35th) for x2Feat...")
            val t025Start = System.currentTimeMillis()
            val quarterBmp = Bitmap.createScaledBitmap(scaledBmp, 384, 384, true)
            patchFloatBuffer.rewind()
            val qOut = mod1.forward(EValue.from(Tensor.fromBlob(preprocess(quarterBmp, true), longArrayOf(1, 3, 384, 384))))
            if (qOut.isEmpty()) {
                LogUtil.e(TAG, "Part1 returned no outputs for 0.25x patch")
                quarterBmp.recycle(); mod1.destroy(); mod2.destroy()
                return@withContext null
            }
            val qTokens = qOut[0].toTensor()
            val qFeat = mod2.forward(EValue.from(qTokens))[0].toTensor().dataAsFloatArray
            reshapeToSpatial(qFeat, x2Feat)
            quarterBmp.recycle()
            LogUtil.d(TAG, "0.25x patch done in ${System.currentTimeMillis() - t025Start}ms. Destroying Part1+Part2...")
            mod1.destroy(); mod2.destroy(); System.gc()

            report(0.44f, "Understanding the full picture…", progressCallback)
            val tPart3Start = System.currentTimeMillis()
            // 3. Image Encoder (Part 3)
            LogUtil.d(TAG, "Loading Part3 (image encoder)...")
            val mod3 = Module.load(findFile("sharp_split_part3_int8.pte")!!.absolutePath)
            imageFloatBuffer.rewind()
            val fullImgTensor = Tensor.fromBlob(preprocess(scaledBmp, false), longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong()))
            val imgTokens = mod3.forward(EValue.from(fullImgTensor))[0].toTensor().dataAsFloatArray
            LogUtil.d(TAG, "Part3 done in ${System.currentTimeMillis() - tPart3Start}ms. imgTokens size=${imgTokens.size}")
            mod3.destroy(); System.gc()
            report(0.46f, "Understanding the full picture…", progressCallback)

            report(0.48f, "Adding depth and shape…", progressCallback)
            val tPart4aStart = System.currentTimeMillis()
            // 4. Chunked Part 4a (tokens 512 + 65); sliceArray for buffer safety with Tensor.fromBlob
            LogUtil.d(TAG, "Starting chunked Part4 decoder...")
            val out512 = runDecoderChunk("sharp_split_part4a_chunk_512.pte", imgTokens.sliceArray(0 until 512 * 1024), 512)
            report(0.49f, "Adding depth and shape…", progressCallback)
            val out65 = runDecoderChunk("sharp_split_part4a_chunk_65.pte", imgTokens.sliceArray(512 * 1024 until 577 * 1024), 65)
            val combinedTokens = out512 + out65
            LogUtil.d(TAG, "Part4a chunks done in ${System.currentTimeMillis() - tPart4aStart}ms. combinedTokens size=${combinedTokens.size}. Loading Part4b...")
            val tPart4bStart = System.currentTimeMillis()

            report(0.50f, "Adding the finishing touches…", progressCallback)
            val part4bThreads = part4bThreadCount()
            var gaussianData: FloatArray? = null

            if (useChunkedPart4bTiled16Files && (hasAll16TileFiles || part4bTileFullFile != null)) {
                val tileModelFile = if (hasAll16TileFiles) part4bTileFiles[0]!! else part4bTileFullFile!!
                val xLowresH = 24
                val xLowresW = 24
                val xLowresC = 1024
                val xLowres = FloatArray(xLowresC * xLowresH * xLowresW)
                for (row in 0 until xLowresH) {
                    for (col in 0 until xLowresW) {
                        val srcBase = (1 + row * xLowresW + col) * xLowresC
                        for (ch in 0 until xLowresC) {
                            xLowres[ch * xLowresH * xLowresW + row * xLowresW + col] = combinedTokens[srcBase + ch]
                        }
                    }
                }
                val fullImgData = fullImgTensor.dataAsFloatArray
                if (NATIVE_TILES_AVAILABLE) {
                    try {
                        report(0.51f, "Generating 3D room (native C++)…", progressCallback)
                        gaussianData = runTiledPart4bNative(
                            tileModelFile.absolutePath,
                            fullImgData, latent0, latent1, x0Feat, x1Feat, x2Feat, xLowres,
                            part4bThreads, PARALLEL_TILES, swapNdcXY
                        )
                        if (gaussianData != null) {
                            val tiledMs = System.currentTimeMillis() - tPart4bStart
                            LogUtil.d(TAG, "[TIMING] Part4b tiled (native C++): ${tiledMs}ms, ${gaussianData.size / 14} Gaussians")
                        }
                    } catch (e: Throwable) {
                        LogUtil.e(TAG, "Native tile pipeline failed, falling back to Kotlin tile loop: ${e.message}", e)
                    }
                }
                // When native didn't produce output (not loaded or threw), run 16-tile path in Kotlin
                if (gaussianData == null) {
                    LogUtil.d(TAG, "Using Kotlin tile loop fallback (native library not loaded or failed)")
                    val grid = PART4B_TILED_GRID
                    val imgC = 3
                    val imgH = IMAGE_SIZE
                    val imgW = IMAGE_SIZE
                    val imgTileH = imgH / grid
                    val imgTileW = imgW / grid
                    val tCropStart = System.currentTimeMillis()
                    val allTileInputs = Array(PART4B_TILED_NUM) { tileIdx ->
                        val tileRow = tileIdx / grid
                        val tileCol = tileIdx % grid
                        arrayOf(
                            EValue.from(Tensor.fromBlob(cropTileNCHW(fullImgData, imgC, imgH, imgW, tileRow, tileCol), longArrayOf(1, 3, imgTileH.toLong(), imgTileW.toLong()))),
                            EValue.from(Tensor.fromBlob(cropTileNCHW(latent0, 1024, 96, 96, tileRow, tileCol), longArrayOf(1, 1024, 24, 24))),
                            EValue.from(Tensor.fromBlob(cropTileNCHW(latent1, 1024, 96, 96, tileRow, tileCol), longArrayOf(1, 1024, 24, 24))),
                            EValue.from(Tensor.fromBlob(cropTileNCHW(x0Feat, 1024, 96, 96, tileRow, tileCol), longArrayOf(1, 1024, 24, 24))),
                            EValue.from(Tensor.fromBlob(cropTileNCHW(x1Feat, 1024, 48, 48, tileRow, tileCol), longArrayOf(1, 1024, 12, 12))),
                            EValue.from(Tensor.fromBlob(cropTileNCHW(x2Feat, 1024, 24, 24, tileRow, tileCol), longArrayOf(1, 1024, 6, 6))),
                            EValue.from(Tensor.fromBlob(cropTileNCHW(xLowres, xLowresC, xLowresH, xLowresW, tileRow, tileCol), longArrayOf(1, 1024, 6, 6)))
                        )
                    }
                    LogUtil.d(TAG, "Pre-computed 16 tile crops in ${System.currentTimeMillis() - tCropStart}ms")
                    val floatsPerTile = PART4B_GAUSSIANS_PER_TILE * 14
                    gaussianData = FloatArray(PART4B_TILED_NUM * floatsPerTile)
                    val modTile = Module.load(tileModelFile.absolutePath, Module.LOAD_MODE_MMAP, part4bThreads)
                    for (tileIdx in 0 until PART4B_TILED_NUM) {
                        val tTile = System.currentTimeMillis()
                        val tileOut = modTile.forward(*allTileInputs[tileIdx])
                        val packedTile = tileOut[0].toTensor().dataAsFloatArray.copyOf()
                        correctNDC(packedTile, tileIdx / grid, tileIdx % grid, swapNdcXY)
                        System.arraycopy(packedTile, 0, gaussianData!!, tileIdx * floatsPerTile, packedTile.size)
                        LogUtil.d(TAG, "Part4b tile $tileIdx: ${packedTile.size / 14} Gaussians, ${System.currentTimeMillis() - tTile}ms")
                        report(0.50f + 0.40f * (tileIdx + 1) / PART4B_TILED_NUM, "Generating 3D room (tile ${tileIdx + 1}/$PART4B_TILED_NUM)…", progressCallback)
                    }
                    modTile.destroy()
                    LogUtil.d(TAG, "[TIMING] Part4b tiled (Kotlin fallback): ${System.currentTimeMillis() - tPart4bStart}ms, ${PART4B_TILED_NUM * PART4B_GAUSSIANS_PER_TILE} Gaussians")
                }
            }

            if (gaussianData == null) {
                val mod4b = Module.load(
                    part4bFile!!.absolutePath,
                    Module.LOAD_MODE_MMAP,
                    part4bThreads
                )
                LogUtil.d(TAG, "Part4b forward: 7 inputs tokens, img, latent0, latent1, x0Feat, x1Feat, x2Feat")
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
                gaussianData = finalParams.dataAsFloatArray.copyOf()
                mod4b.destroy()
                val part4bMs = System.currentTimeMillis() - tPart4bStart
                LogUtil.d(TAG, "[TIMING] Part4b forward: ${part4bMs}ms")
            }

            report(0.92f, "Saving your 3D room…", progressCallback)
            val tPlyStart = System.currentTimeMillis()
            LogUtil.d(TAG, "[PLY] Part4b done. Writing PLY with raw coordinates (center-crop input = 1:1, no aspect scale)")
            val result = writePly(gaussianData, progressCallback, isPortrait)
            val plyMs = System.currentTimeMillis() - tPlyStart
            LogUtil.d(TAG, "[TIMING] writePly: ${plyMs}ms")
            scaledBmp.recycle()
            report(1f, "Your room is ready!", progressCallback)
            val totalMs = System.currentTimeMillis() - pipelineStartMs
            LogUtil.d(TAG, "[ASPECT] inference complete | Gaussians=${result.gaussianCount} | PLY bbox room=${result.roomWidth}x${result.roomHeight}x${result.roomDepth} m")
            LogUtil.d(TAG, "[TIMING] TOTAL pipeline: ${totalMs}ms (${totalMs / 1000f}s)")
            return@withContext result
        }
    }

    private fun runDecoderChunk(name: String, data: FloatArray, count: Int): FloatArray {
        LogUtil.d(TAG, "runDecoderChunk: $name count=$count dataLen=${data.size}")
        val mod = Module.load(findFile(name)!!.absolutePath)
        patchFloatBuffer.rewind()
        val inputTensor = Tensor.fromBlob(data, longArrayOf(1, count.toLong(), 1024))
        val output = mod.forward(EValue.from(inputTensor))
        if (output.isEmpty()) {
            LogUtil.e(TAG, "runDecoderChunk: $name returned no outputs")
            mod.destroy()
            return FloatArray(0)
        }
        val out = output[0].toTensor().dataAsFloatArray
        LogUtil.d(TAG, "runDecoderChunk: $name outputLen=${out.size}")
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

    /**
     * Write PLY from model output. Uses raw coordinates (no aspect scaling): model trained on square input
     * expects 1:1 coordinate space. Only y,z are negated for our viewer convention.
     * See Ultralytics: apply aspectRatio only when mapping to non-square frustum; else normalization issues cause jagged output.
     */
    private fun writePly(
        params: FloatArray,
        progressCallback: ((Float, String) -> Unit)?,
        isPortrait: Boolean = false
    ): StreamingResult {
        val count = params.size / PARAMS_PER_GAUSSIAN
        LogUtil.d(TAG, "[PLY] writePly: count=$count isPortrait=$isPortrait (raw x,y,z; y,z negated)")
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
                val rawX = params[off]
                val rawY = params[off + 1]
                val rawZ = params[off + 2]
                val x = rawX
                val y = -rawY
                val z = -rawZ
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
        val roomW = maxX - minX
        val roomH = maxY - minY
        val roomD = maxZ - minZ
        val centerX = (minX + maxX) * 0.5f
        val centerY = (minY + maxY) * 0.5f
        val centerZ = (minZ + maxZ) * 0.5f
        LogUtil.d(TAG, "[PLY] saved bbox: roomWidth=$roomW roomHeight=$roomH roomDepth=$roomD (raw coords)")
        if (roomW > 50f || roomH > 50f || roomD > 50f || roomW < 0.1f || roomH < 0.1f || roomD < 0.1f) {
            LogUtil.w(TAG, "[PLY] bbox may indicate scale/precision issue (expected room ~2–15 m)")
        }
        return StreamingResult(plyFile, plyFile, count, roomW, roomH, roomD, centerX, centerY, centerZ)
    }
}

