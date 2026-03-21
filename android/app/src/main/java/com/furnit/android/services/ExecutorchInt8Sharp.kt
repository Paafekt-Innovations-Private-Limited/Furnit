package com.furnit.android.services

// BUILD_ID_TERMINAL_2025
import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import com.furnit.android.BuildConfig
import com.furnit.android.utils.LogUtil
import com.furnit.android.utils.Part1OnlyTest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
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
 *
 * **Room generation (`inferStreaming`) runs the native C++ full pipeline only** (`runFullPipelineInt8Native` in
 * `sharp_executorch_full.cpp`). There is intentionally no Kotlin Part1–Part4 fallback here—keep all graph work in C++.
 * Part1+2 always use the full patch grid (25× + 9× + 0.25× = 35 encoder passes); Vulkan/GPU work happens inside native
 * `forward()` on the IO dispatcher—Kotlin `async`/`delay` are not required for that path.
 *
 * Input: center-crop to square (min(w,h)) then resize to 1536x1536. We do not pad gray to make a bigger square
 * (e.g. 2x4 -> 4x4 with gray on sides): that was tried (114-gray letterbox) and caused jagged output—ViT sees
 * the content/padding edge as OOD; INT8 amplifies it. Raw PLY coords; no aspect scale.
 *
 * Export note: if jagged output persists, consider FP16 for position/scale decoder heads
 * (see Ultralytics deployment practices; INT8 on scales/rotations can cause severe artifacts).
 */
class ExecutorchInt8Sharp private constructor(private val context: Context) {

    /** Set by inferStreaming before native call; cleared after. Used by reportProgressFromNative (JNI callback). */
    @Volatile
    private var currentProgressCallback: ((Float, String) -> Unit)? = null

    /** Called from native (JNI) during runFullPipelineInt8Native to report progress so UI does not appear stuck at 20%. */
    fun reportProgressFromNative(progress: Float, message: String) {
        val cb = currentProgressCallback ?: return
        android.os.Handler(android.os.Looper.getMainLooper()).post { cb.invoke(progress.coerceIn(0f, 1f), message) }
    }

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
        private const val IMAGE_SIZE_1280 = 1280
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

        // Recommended default splat limit for ExecuTorch INT8 full pipeline
        private const val MAX_GAUSSIANS_LIMIT = 500_000

        private const val MODEL_FILENAME = "sharp_full_vulkan.pte"
        private val SPLIT_FILENAMES = arrayOf(
            "sharp_split_part1_vulkan_fp32.pte",
            "sharp_split_part2_vulkan_fp32.pte",
            "sharp_split_part3_vulkan_fp32.pte",
            "sharp_split_part4b_vulkan.pte"
        )
        /** Names of .pte files that may be packaged in assets/models/ for testing. */
        private val ASSET_MODEL_FILENAMES = arrayOf(
            "sharp_split_part1_vulkan_fp32.pte",
            "sharp_split_part2_vulkan_fp32.pte",
            "sharp_split_part3_vulkan_fp32.pte",
            "sharp_split_part3_1280_vulkan_fp32.pte",
            "sharp_split_part1_vulkan_fp16.pte",
            "sharp_split_part2_vulkan_fp16.pte",
            "sharp_split_part3_vulkan_fp16.pte",
            "sharp_split_part3_1280_vulkan_fp16.pte",
            "sharp_split_part4_1280_vulkan_fp32.pte",
            "sharp_split_part4_1280_vulkan_fp16.pte",
            "sharp_split_part4a_chunk_512_vulkan.pte",
            "sharp_split_part4a_chunk_65_vulkan.pte",
            "sharp_split_part4b_vulkan.pte",
            "sharp_split_part4b_tile_b2.pte",
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
        /** External + internal storage subdir per Gradle flavor: keeps CPU vs Vulkan .pte separate on device. */
        const val MODELS_SUBDIR_CPU = "models_cpu"
        const val MODELS_SUBDIR_VULKAN = "models_vulkan"
        /** Legacy single folder (still searched as fallback). */
        const val MODELS_SUBDIR_LEGACY = "models"

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

        private fun orderedVulkanPrecisionNames(stem: String, preferFp16: Boolean): Array<String> =
            if (preferFp16) {
                arrayOf("${stem}_vulkan_fp16.pte", "${stem}_vulkan_fp32.pte")
            } else {
                arrayOf("${stem}_vulkan_fp32.pte", "${stem}_vulkan_fp16.pte")
            }

        /** Whether the native 16-tile Part4b library loaded successfully. When false, Kotlin/single path is used. */
        @JvmField
        var NATIVE_TILES_AVAILABLE: Boolean = false

        /** Whether the native full INT8 pipeline library loaded (Part1–4b in C++, single Part4b). When true and pref on, inferStreaming tries C++ first. */
        @JvmField
        var NATIVE_FULL_AVAILABLE: Boolean = false

        init {
            // Load libexecutorch before JNI so Vulkan (and other) backends register; then sharp_* link same runtime.
            try {
                com.furnit.android.utils.ExecutorchNativeLoader.loadForJavaModule()
            } catch (e: UnsatisfiedLinkError) {
                LogUtil.d(TAG, "ExecuTorch native preload failed: ${e.message}")
            }
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
    /** Matches etVulkan vs etCpu APK: `files/models_vulkan` or `files/models_cpu` (+ scoped external storage). */
    private val executorchModelsSubdir: String =
        if (BuildConfig.EXECUTORCH_USE_VULKAN_AAR) MODELS_SUBDIR_VULKAN else MODELS_SUBDIR_CPU
    /** External dir for adb push: try type first, then base + subdir (some devices return null for type). */
    private val externalModelsDir: File?
        get() = context.getExternalFilesDir(executorchModelsSubdir)
            ?: context.getExternalFilesDir(null)?.let { File(it, executorchModelsSubdir).takeIf { d -> d.exists() || d.mkdirs() } }
    private val modelsDir by lazy {
        externalModelsDir ?: File(context.filesDir, executorchModelsSubdir)
    }
    private val internalModelsDir by lazy {
        File(context.filesDir, executorchModelsSubdir).also { it.mkdirs() }
    }

    private val plyBatch = ByteBuffer.allocateDirect(BYTES_PER_VERTEX * PLY_BATCH_SIZE).apply { order(ByteOrder.LITTLE_ENDIAN) }
    private val zeroSHBuffer = ByteBuffer.allocateDirect(45 * 4).apply { order(ByteOrder.LITTLE_ENDIAN) }
    private val patchPixelBuffer = ByteBuffer.allocateDirect(PATCH_SIZE * PATCH_SIZE * 4).order(ByteOrder.nativeOrder())
    private val patchFloatBuffer = ByteBuffer.allocateDirect(3 * PATCH_SIZE * PATCH_SIZE * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val imagePixelBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 4).order(ByteOrder.nativeOrder())
    private val imageFloatBuffer = ByteBuffer.allocateDirect(3 * INPUT_SIZE * INPUT_SIZE * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()

    /** Pinned float[] for ExecuTorch: fromBlob(FloatBuffer) can cause native ptr abort; use float[] only. */
    private val patchInputFloats = FloatArray(3 * PATCH_SIZE * PATCH_SIZE)
    private val tokenInputFloats = FloatArray(577 * FEATURE_DIM)
    private val fullImageInputFloats = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE)

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
            // First sync pushed models into internal storage so native preload sees the final set.
            syncExternalSharpSplitPteToInternal()
            syncExternalCpuSharpSplitPteToInternal()
            if (NATIVE_FULL_AVAILABLE) {
                val prefs = context.getSharedPreferences("furnit_prefs", Context.MODE_PRIVATE)
                val useCpuStable = prefs.getBoolean("executorch_int8_use_cpu_stable", false)
                val preferVulkanFp16Requested = prefs.getBoolean("executorch_vulkan_prefer_fp16", false)
                val preferVulkanFp16 = effectivePreferVulkanFp16(preferVulkanFp16Requested)
                if (preferVulkanFp16Requested && !preferVulkanFp16) {
                    LogUtil.w(
                        TAG,
                        "Ignoring 'Prefer Vulkan FP16 models' because this APK uses executorch-android-vulkan AAR " +
                            "and the required FP16 shaders are not bundled. Falling back to FP32-safe Vulkan models."
                    )
                }
                val part12OnCpuRequested = prefs.getBoolean("executorch_int8_part12_on_cpu", false)
                val effectivePart12OnCpu =
                    !useCpuStable && (part12OnCpuRequested || hasCpuPart12SidecarModels())
                val cppModelDir = if (useCpuStable) {
                    findSinglePart4bCpuPte()?.parent
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_B2)?.parent
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_B4)?.parent
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_00)?.parent
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_FULL)?.parent
                        ?: findFile(SharpExecuTorchSplitModelNames.PART1_INT8)?.parent
                        ?: findFile(SharpExecuTorchSplitModelNames.PART1_FP32)?.parent
                } else {
                    // Hybrid Vulkan path keeps Part3/4 under models_vulkan; native CPU Part1+2 loader
                    // resolves sibling models_cpu automatically when effectivePart12OnCpu=true.
                    findVulkanPart1Pte(preferVulkanFp16)?.parent
                        ?: findFile("sharp_split_part4b_vulkan.pte")?.parent
                } ?: modelsDir?.absolutePath ?: internalModelsDir.absolutePath
                val useVulkanForPart12 = !useCpuStable && !effectivePart12OnCpu
                val preloaded = preloadCppModules(cppModelDir, useVulkanForPart12)
                LogUtil.d(TAG, "[C++ FULL] Preload Part1+Part2 cache: ${if (preloaded) "OK" else "failed"} (dir=$cppModelDir useVulkan=${!useCpuStable} part12OnCpu=$effectivePart12OnCpu)")
                if (!useCpuStable && effectivePart12OnCpu) {
                    LogUtil.i(TAG, "[C++ FULL] Hybrid mode forced: Part1+2 from models_cpu INT8, Part3/4 from models_vulkan")
                }
                LogUtil.i(
                    TAG,
                    "BUILD_CONFIG EXECUTORCH_USE_VULKAN_AAR=${BuildConfig.EXECUTORCH_USE_VULKAN_AAR} " +
                        "(false = etCpu flavor / XNNPACK AAR; true = etVulkan). settings_cpu_stable=$useCpuStable"
                )
                LogUtil.i(
                    TAG,
                    "ExecuTorch model roots: internal=${internalModelsDir.absolutePath} external=${modelsDir.absolutePath} " +
                        "(push CPU .pte to .../files/$MODELS_SUBDIR_CPU/, Vulkan to .../files/$MODELS_SUBDIR_VULKAN/)"
                )
                if (!BuildConfig.EXECUTORCH_USE_VULKAN_AAR && !useCpuStable) {
                    LogUtil.w(
                        TAG,
                        "APK ExecuTorch is XNNPACK-only but Settings use Vulkan model layout — Vulkan .pte may fail. " +
                            "Either enable CPU ExecuTorch INT8 in Settings or install the etVulkan APK: ./gradlew :app:assembleEtVulkanDebug"
                    )
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
            LogUtil.d(TAG, "Part4b tile models: ${tilesCopied}/17 from assets (add tile .pte to executorch_models or sharp_vulkan_only and rebuild for tiled path)")
        } else if (tilesCopied == 17) {
            LogUtil.d(TAG, "Part4b tile models ready from assets (16 tiles + tile_full)")
        }
        syncExternalSharpSplitPteToInternal()
        syncExternalCpuSharpSplitPteToInternal()
    }

    /**
     * Copy `sharp_split*.pte` from external (adb push target) into internal storage.
     * Call from preloadSharpModels() so opening the SHARP screen syncs without running inference.
     */
    fun syncModelsFromExternal() {
        syncExternalSharpSplitPteToInternal()
        syncExternalCpuSharpSplitPteToInternal()
    }

    /**
     * Copy `sharp_split*.pte` from flavor-scoped external dir (adb push target) into internal storage
     * so mmap loads are fast. etCpu: .../files/models_cpu/ ; etVulkan: .../files/models_vulkan/ .
     */
    private fun syncExternalSharpSplitPteToInternal() {
        val externalDir = externalModelsDir ?: return
        if (externalDir.absolutePath == internalModelsDir.absolutePath) return
        val sources = externalDir.listFiles() ?: return
        LogUtil.d(TAG, "sync: external=${externalDir.absolutePath} files=${sources.count { it.isFile && it.name.startsWith("sharp_split") && it.name.endsWith(".pte", ignoreCase = true) }}")
        val externalSharpSplitNames = HashSet<String>()
        for (src in sources) {
            if (!src.isFile || !src.name.endsWith(".pte", ignoreCase = true)) continue
            if (!src.name.startsWith("sharp_split")) continue
            externalSharpSplitNames.add(src.name)
        }
        var anyCopied = false
        for (src in sources) {
            if (!src.isFile || !src.name.endsWith(".pte", ignoreCase = true)) continue
            if (!src.name.startsWith("sharp_split")) continue
            val dest = File(internalModelsDir, src.name)
            if (dest.exists() && dest.length() == src.length()) continue
            try {
                src.copyTo(dest, overwrite = true)
                LogUtil.d(TAG, "Synced ${src.name} from external to ${dest.absolutePath}")
                anyCopied = true
            } catch (e: Exception) {
                LogUtil.e(TAG, "Sync ${src.name} from external failed: ${e.message}")
            }
        }
        // After a v2-only push, external may have only 6 files — drop stale internal sharp_split*.pte (e.g. old part4b.pte).
        var anyPruned = false
        if (externalSharpSplitNames.isNotEmpty()) {
            internalModelsDir.listFiles()?.forEach { internalFile ->
                if (!internalFile.isFile || !internalFile.name.endsWith(".pte", ignoreCase = true)) return@forEach
                if (!internalFile.name.startsWith("sharp_split")) return@forEach
                if (externalSharpSplitNames.contains(internalFile.name)) return@forEach
                try {
                    if (internalFile.delete()) {
                        LogUtil.i(TAG, "Pruned internal models_cpu (not on external): ${internalFile.name}")
                        anyPruned = true
                    }
                } catch (e: Exception) {
                    LogUtil.w(TAG, "Could not prune ${internalFile.name}: ${e.message}")
                }
            }
        } else {
            LogUtil.d(TAG, "sync: no sharp_split*.pte on external — skip internal prune")
        }
        if ((anyCopied || anyPruned) && NATIVE_FULL_AVAILABLE) {
            try {
                releaseCppModules()
                LogUtil.d(TAG, "Released Part1+Part2 cache after external sync / prune")
            } catch (_: Throwable) { }
        }
    }

    private fun syncExternalCpuSharpSplitPteToInternal() {
        val externalCpuDir = context.getExternalFilesDir(MODELS_SUBDIR_CPU)
            ?: context.getExternalFilesDir(null)?.let { File(it, MODELS_SUBDIR_CPU).takeIf { d -> d.exists() || d.mkdirs() } }
            ?: return
        val internalCpuDir = File(context.filesDir, MODELS_SUBDIR_CPU).also { it.mkdirs() }
        if (externalCpuDir.absolutePath == internalCpuDir.absolutePath) return
        val sources = externalCpuDir.listFiles() ?: return
        val cpuPtes = sources.count { it.isFile && it.name.startsWith("sharp_split") && it.name.endsWith(".pte", ignoreCase = true) }
        if (cpuPtes <= 0) return
        LogUtil.d(TAG, "sync cpu sidecars: external=${externalCpuDir.absolutePath} files=$cpuPtes")
        var anyCopied = false
        for (src in sources) {
            if (!src.isFile || !src.name.endsWith(".pte", ignoreCase = true)) continue
            if (!src.name.startsWith("sharp_split")) continue
            val dest = File(internalCpuDir, src.name)
            if (dest.exists() && dest.length() == src.length()) continue
            try {
                src.copyTo(dest, overwrite = true)
                LogUtil.d(TAG, "Synced CPU sidecar ${src.name} from external to ${dest.absolutePath}")
                anyCopied = true
            } catch (e: Exception) {
                LogUtil.w(TAG, "Sync CPU sidecar ${src.name} from external failed: ${e.message}")
            }
        }
        if (anyCopied && NATIVE_FULL_AVAILABLE) {
            try {
                releaseCppModules()
                LogUtil.d(TAG, "Released Part1+Part2 cache after CPU sidecar sync")
            } catch (_: Throwable) { }
        }
    }

    /**
     * Part4b thread count: fixed 2 so behavior matches 8 GB phone and does not OOM on any device
     * (12 GB and others). More threads can be re-enabled later for simulator/testing.
     */
    private fun part4bThreadCount(): Int = 2

    /**
     * Resolve a model file. Flavor dirs first: [internalModelsDir] / [modelsDir] (`models_cpu` or `models_vulkan`).
     *
     * **etCpu + `sharp_split*.pte`:** legacy `files/models` is **not** searched. Otherwise an old
     * `sharp_split_part4b.pte` (e.g. Vulkan-delegated) wins over `models_cpu` and breaks XNNPACK (`NotFound` /
     * "VulkanBackend is not registered"). Push split `.pte` only under `models_cpu`.
     */
    private fun findFile(filename: String): File? {
        File(internalModelsDir, filename).takeIf { it.exists() && it.length() > 0L }?.let { return it }
        File(modelsDir, filename).takeIf { it.exists() && it.length() > 0L }?.let { return it }
        val skipLegacySplitOnEtCpu =
            filename.startsWith("sharp_split") && !BuildConfig.EXECUTORCH_USE_VULKAN_AAR
        if (skipLegacySplitOnEtCpu) {
            LogUtil.d(
                TAG,
                "findFile($filename): etCpu — skipping legacy ${MODELS_SUBDIR_LEGACY}/ " +
                    "(use .../files/$MODELS_SUBDIR_CPU/ on device)"
            )
            return null
        }
        File(context.filesDir, MODELS_SUBDIR_LEGACY).resolve(filename)
            .takeIf { it.exists() && it.length() > 0L }?.let { return it }
        context.getExternalFilesDir(MODELS_SUBDIR_LEGACY)?.let { legacyExt ->
            File(legacyExt, filename).takeIf { it.exists() && it.length() > 0L }?.let { return it }
        }
        return null
    }

    /** CPU portable/INT8 Part1+2 sidecar models for hybrid Vulkan runs live under files/models_cpu. */
    private fun findCpuSidecarFile(filename: String): File? {
        val internalCpuDir = File(context.filesDir, MODELS_SUBDIR_CPU)
        File(internalCpuDir, filename).takeIf { it.exists() && it.length() > 0L }?.let { return it }
        val externalCpuDir = context.getExternalFilesDir(MODELS_SUBDIR_CPU)
            ?: context.getExternalFilesDir(null)?.let { File(it, MODELS_SUBDIR_CPU).takeIf { d -> d.exists() || d.mkdirs() } }
        externalCpuDir?.let { dir ->
            File(dir, filename).takeIf { it.exists() && it.length() > 0L }?.let { return it }
        }
        return null
    }

    private fun hasCpuPart12SidecarModels(): Boolean =
        (findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART1_INT8) != null ||
            findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART1_FP32) != null ||
            findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART1_FP16) != null) &&
            (findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART2_INT8) != null ||
                findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART2_FP32) != null ||
                findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART2_FP16) != null)

    private fun effectivePreferVulkanFp16(requested: Boolean): Boolean =
        requested && !BuildConfig.EXECUTORCH_USE_VULKAN_AAR

    /** Default order is FP32-first; low-memory Vulkan mode flips this to FP16-first when such exports exist. */
    private fun findVulkanPart1Pte(preferFp16: Boolean = false): File? =
        orderedVulkanPrecisionNames("sharp_split_part1", preferFp16).firstNotNullOfOrNull(::findFile)

    private fun findVulkanPart2Pte(preferFp16: Boolean = false): File? =
        orderedVulkanPrecisionNames("sharp_split_part2", preferFp16).firstNotNullOfOrNull(::findFile)

    private fun findVulkanPart3Pte(preferFp16: Boolean = false): File? =
        orderedVulkanPrecisionNames("sharp_split_part3", preferFp16).firstNotNullOfOrNull(::findFile)

    private fun findVulkan1280Part3Pte(preferFp16: Boolean = false): File? =
        orderedVulkanPrecisionNames("sharp_split_part3_1280", preferFp16).firstNotNullOfOrNull(::findFile)

    private fun findVulkan1280Part4Pte(preferFp16: Boolean = false): File? =
        orderedVulkanPrecisionNames("sharp_split_part4_1280", preferFp16).firstNotNullOfOrNull(::findFile)

    private fun hasTrue1280VulkanModelSet(preferFp16: Boolean = false): Boolean =
        findVulkan1280Part3Pte(preferFp16) != null && findVulkan1280Part4Pte(preferFp16) != null

    /** Full-image Part4b single decoder: INT8, else FP16, else FP32. Null if only tiled `.pte` exist. */
    private fun findSinglePart4bCpuPte(): File? =
        findFile(SharpExecuTorchSplitModelNames.PART4B_INT8)
            ?: findFile(SharpExecuTorchSplitModelNames.PART4B_FP16)
            ?: findFile(SharpExecuTorchSplitModelNames.PART4B_FP32)

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
     *
     * @param maxGaussians Upper bound on number of Gaussians to keep (0 = unlimited).
     * @param preferSinglePart4b When true, forces single Part4b decoder instead of tiled paths for stability.
     * @param useVulkan When true, allows C++ pipeline to prefer Vulkan-backed ExecuTorch load modes when available.
     * Part1 patch limits / chunking: native still takes ints; Kotlin always passes 0 = full 25+9 grid, no chunking.
     */
    /**
     * @param etdumpOutputPath When non-null (e.g. from "Record ETDump on next run" in Settings),
     * the Vulkan Part4b run records an ExecuTorch ETDump to this path for inspection with
     * inspector_cli.py. Requires ExecuTorch built with EXECUTORCH_ENABLE_EVENT_TRACER=ON.
     */
    private external fun runFullPipelineInt8Native(
        modelDirPath: String,
        imageNCHW: FloatArray,
        maxGaussians: Int,
        preferSinglePart4b: Boolean,
        useVulkan: Boolean,
        part12OnCpu: Boolean,
        part12ForceSinglePatch: Boolean,
        part12_25Only: Boolean,
        part1MaxPatches1x: Int,
        part1MaxPatches05x: Int,
        part12Chunk1x: Int,
        part12Chunk05x: Int,
        part12YieldMsBetweenChunks: Int,
        swapTileNdcXY: Boolean,
        progressReporter: ExecutorchInt8Sharp,
        etdumpOutputPath: String?
    ): FloatArray?

    /**
     * Double protection: validate JNI FloatArray and catch crashes (Vulkan AOT metadata corruption
     * can produce bad size / NaN). Returns null if invalid so caller can fall back to Kotlin.
     */
    private fun safeJniFloatArrayResult(block: () -> FloatArray?): FloatArray? {
        return try {
            val result = block()
            when {
                result == null -> {
                    LogUtil.e(TAG, "JNI result: null")
                    null
                }
                result.isEmpty() -> {
                    LogUtil.e(TAG, "JNI result: empty")
                    null
                }
                result.size % 14 != 0 -> {
                    LogUtil.e(TAG, "JNI result invalid: size=${result.size} not divisible by 14")
                    null
                }
                result.size > 50_000_000 -> {
                    LogUtil.e(TAG, "JNI result invalid: size=${result.size} too large")
                    null
                }
                result.any { it.isNaN() || it.isInfinite() } -> {
                    LogUtil.e(TAG, "JNI result invalid: contains NaN or Infinite")
                    null
                }
                else -> result
            }
        } catch (e: Throwable) {
            LogUtil.e(TAG, "JNI crash caught: ${e.message}", e)
            null
        }
    }

    /** Warm-start: preload Part1+Part2. Vulkan path = Vulkan only (no CPU fallback). */
    private external fun preloadCppModules(modelDirPath: String, useVulkanForPart12: Boolean): Boolean

    /** Release native singleton Part1+Part2 cache and aligned workspace buffers. */
    private external fun releaseCppModules()

    /**
     * Release caches that are useful for warmup/bench but waste RAM for the full room-generation path.
     * This drops the Java Part1 warmup session and the native preload cache so the heavy pipeline starts clean.
     */
    private fun releasePreInferenceCaches(reason: String) {
        val beforeMb = getAvailMemBytes() / (1024 * 1024)
        try {
            Part1OnlyTest.releaseCachedPart1Module()
        } catch (_: Throwable) { }
        try {
            releaseCppModules()
        } catch (_: Throwable) { }
        System.gc()
        val afterMb = getAvailMemBytes() / (1024 * 1024)
        LogUtil.d(TAG, "[MEMORY] Released pre-inference caches reason=$reason avail_before=${beforeMb}MB avail_after=${afterMb}MB")
    }

    /** Release native caches (Part1+Part2, workspace) and Java Part1 warmup cache. */
    fun releaseNativeCaches() {
        try { Part1OnlyTest.releaseCachedPart1Module() } catch (_: Throwable) { }
        try { releaseCppModules() } catch (_: Throwable) { }
    }

    /** Current memory info for logging; helps diagnose OOM / CoroutineScheduler kills. */
    private fun getMemoryInfo(): String {
        return try {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager ?: return "N/A"
            val mem = android.app.ActivityManager.MemoryInfo()
            am.getMemoryInfo(mem)
            "avail=${mem.availMem / (1024 * 1024)}MB total=${mem.totalMem / (1024 * 1024)}MB low=${mem.lowMemory}"
        } catch (_: Throwable) { "N/A" }
    }

    /** If "Record ETDump on next run" is enabled in Settings, returns output path and clears the pref; otherwise null. */
    private fun getAndClearEtdumpOutputPath(): String? {
        val prefs = context.getSharedPreferences("furnit_prefs", Context.MODE_PRIVATE)
        if (!prefs.getBoolean("executorch_record_etdump_next_run", false)) return null
        prefs.edit().putBoolean("executorch_record_etdump_next_run", false).apply()
        val dir = context.getExternalFilesDir(null) ?: context.filesDir
        return File(dir, "sharp_part4b.etdp").absolutePath
    }

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

    /** Minimum free RAM (bytes) to attempt C++ full pipeline; below this use Kotlin path to avoid OOM. */
    private fun getAvailMemBytes(): Long {
        return try {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager ?: return Long.MAX_VALUE
            val mem = android.app.ActivityManager.MemoryInfo()
            am.getMemoryInfo(mem)
            mem.availMem
        } catch (_: Throwable) { Long.MAX_VALUE }
    }

    suspend fun inferStreaming(
        bitmap: Bitmap,
        progressCallback: ((Float, String) -> Unit)? = null,
        useVulkan: Boolean = true,
        maxGaussians: Int = MAX_GAUSSIANS_LIMIT
    ): StreamingResult? = withContext(Dispatchers.IO) {
        mutex.withLock {
            if (!isInitialized) return@withContext null
            LogUtil.d(TAG, "[MEMORY] ${getMemoryInfo()}")
            ensureModelsFromAssets()

            val prefs = context.getSharedPreferences("furnit_prefs", Context.MODE_PRIVATE)
            val useCpuStable = prefs.getBoolean("executorch_int8_use_cpu_stable", false)
            val preferVulkanFp16Requested = prefs.getBoolean("executorch_vulkan_prefer_fp16", false)
            val preferVulkanFp16 = effectivePreferVulkanFp16(preferVulkanFp16Requested)
            if (preferVulkanFp16Requested && !preferVulkanFp16) {
                LogUtil.w(
                    TAG,
                    "Ignoring 'Prefer Vulkan FP16 models' because this APK uses executorch-android-vulkan AAR " +
                        "and the required FP16 shaders are not bundled. Falling back to FP32-safe Vulkan models."
                )
            }
            val part12OnCpuRequested = prefs.getBoolean("executorch_int8_part12_on_cpu", false)
            val part12OnCpu = !useCpuStable && (part12OnCpuRequested || hasCpuPart12SidecarModels())
            // Hybrid Vulkan path: Part1+2 can run on CPU from models_cpu while Part3/4 stay on Vulkan from models_vulkan.
            fun hasPart1() = if (useCpuStable) {
                findFile(SharpExecuTorchSplitModelNames.PART1_INT8) != null ||
                    findFile(SharpExecuTorchSplitModelNames.PART1_FP32) != null ||
                    findFile(SharpExecuTorchSplitModelNames.PART1_FP16) != null
            } else if (part12OnCpu) {
                findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART1_INT8) != null ||
                    findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART1_FP32) != null ||
                    findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART1_FP16) != null
            } else {
                findVulkanPart1Pte(preferVulkanFp16) != null
            }
            fun hasPart2() = if (useCpuStable) {
                findFile(SharpExecuTorchSplitModelNames.PART2_INT8) != null ||
                    findFile(SharpExecuTorchSplitModelNames.PART2_FP32) != null ||
                    findFile(SharpExecuTorchSplitModelNames.PART2_FP16) != null
            } else if (part12OnCpu) {
                findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART2_INT8) != null ||
                    findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART2_FP32) != null ||
                    findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART2_FP16) != null
            } else {
                findVulkanPart2Pte(preferVulkanFp16) != null
            }
            val allCpu = useCpuStable
            fun hasPart3() = if (useCpuStable) {
                findFile(SharpExecuTorchSplitModelNames.PART3_INT8) != null ||
                    findFile(SharpExecuTorchSplitModelNames.PART3_FP16) != null ||
                    findFile(SharpExecuTorchSplitModelNames.PART3_FP32) != null
            } else {
                findVulkanPart3Pte(preferVulkanFp16) != null
            }
            fun hasPart4a512() = if (allCpu) findFile(SharpExecuTorchSplitModelNames.PART4A_CHUNK_512) != null
            else findFile("sharp_split_part4a_chunk_512_vulkan.pte") != null
            fun hasPart4a65() = if (allCpu) findFile(SharpExecuTorchSplitModelNames.PART4A_CHUNK_65) != null
            else findFile("sharp_split_part4a_chunk_65_vulkan.pte") != null
            val part4bFile = if (useCpuStable) {
                findSinglePart4bCpuPte()
            } else {
                findFile("sharp_split_part4b_vulkan.pte")
            }
            fun hasSplitTile00Part4bVulkan(): Boolean {
                if (useCpuStable) return false
                val n = SharpExecuTorchSplitModelNames
                return findFile(n.PART4B_TILE_00_STAGE_A_VULKAN) != null &&
                    findFile(n.PART4B_TILE_00_INIT_BASE) != null &&
                    findFile(n.PART4B_TILE_00_RAW_HEADS_VULKAN) != null &&
                    findFile(n.PART4B_TILE_00_COMPOSE) != null
            }
            fun hasSplitTileB2Part4bVulkan(): Boolean {
                if (useCpuStable) return false
                val n = SharpExecuTorchSplitModelNames
                return findFile(n.PART4B_TILE_B2_STAGE_A_VULKAN) != null &&
                    findFile(n.PART4B_TILE_B2_INIT_BASE) != null &&
                    findFile(n.PART4B_TILE_B2_RAW_HEADS_VULKAN) != null &&
                    findFile(n.PART4B_TILE_B2_COMPOSE) != null
            }
            // Non-single Part4b: batched tile_b2/tile_b4, split tile_00 / tile_full sequential fallback, or 16× per-tile exports.
            // Vulkan C++ now tries split tile_b2 first when available, then legacy tile_b2/tile_b4,
            // then split tile_00 / tile_full, then single part4b.
            fun hasPart4bNonSingleDecoder(): Boolean {
                val n = SharpExecuTorchSplitModelNames
                return hasSplitTileB2Part4bVulkan() ||
                    hasSplitTile00Part4bVulkan() ||
                    findFile(n.PART4B_TILE_B2) != null ||
                    findFile(n.PART4B_TILE_B4) != null ||
                    findFile(n.PART4B_TILE_00) != null ||
                    findFile(n.PART4B_TILE_FULL) != null
            }
            val hasPart4bTiled = hasPart4bNonSingleDecoder()
            val part4bSatisfied = part4bFile != null || hasPart4bTiled
            // Vulkan: keep tiled Part4b when tiled exports exist to avoid the single-decoder memory peak.
            // CPU stable mode can still force the single decoder because the CPU tiled path is much slower on some devices.
            val preferSinglePart4bExplicit = prefs.getBoolean("executorch_prefer_single_part4b", false)
            val forceTiledPart4bOnVulkan = !useCpuStable && hasPart4bTiled && preferSinglePart4bExplicit
            if (forceTiledPart4bOnVulkan) {
                LogUtil.w(
                    TAG,
                    "Ignoring Prefer single Part4b on Vulkan because tiled Part4b exports are present " +
                        "and the single Vulkan decoder is memory-unstable on this path"
                )
            }
            val preferSinglePart4b = useCpuStable || (!forceTiledPart4bOnVulkan && (preferSinglePart4bExplicit || !hasPart4bTiled))
            // Effective Gaussian cap for full C++ pipeline. 0 = All (unlimited, no pruning).
            var effectiveMaxGaussians = prefs.getInt("executorch_int8_max_gaussians", 500000)
            val availMem = getAvailMemBytes()
            if (availMem < 600L * 1024 * 1024 && (effectiveMaxGaussians <= 0 || effectiveMaxGaussians > 300_000)) {
                val capped = 300_000
                LogUtil.d(TAG, "[MEMORY] Capping maxGaussians to $capped (avail=${availMem / (1024 * 1024)}MB)")
                effectiveMaxGaussians = capped
            }
            val requiredSatisfied = hasPart1() && hasPart2() && hasPart3() && hasPart4a512() && hasPart4a65() && part4bSatisfied
            if (!requiredSatisfied) {
                val missing = mutableListOf<String>()
                if (!hasPart1()) missing.add(if (useCpuStable || part12OnCpu) "part1 portable" else "part1 vulkan")
                if (!hasPart2()) missing.add(if (useCpuStable || part12OnCpu) "part2 portable" else "part2 vulkan")
                if (!hasPart3()) missing.add(if (useCpuStable) "part3 portable" else "part3 vulkan")
                if (!hasPart4a512()) missing.add(if (useCpuStable) "part4a_512" else "part4a_512_vulkan")
                if (!hasPart4a65()) missing.add(if (useCpuStable) "part4a_65" else "part4a_65_vulkan")
                if (!part4bSatisfied) {
                    missing.add(
                        if (useCpuStable) {
                            "part4b_int8 / part4b_fp16 / part4b.pte / part4b_tile_b2 / part4b_tile_b4 / tile_00 / tile_full"
                        } else {
                            "part4b_vulkan.pte / split tile_b2 or tile_00 (stage_a + init_base + raw_heads + compose)"
                        }
                    )
                }
                LogUtil.e(TAG, "ExecuTorch INT8 models missing (${if (useCpuStable) "CPU" else "Vulkan"}): $missing. Push to ${modelsDir?.absolutePath ?: internalModelsDir.absolutePath}")
                return@withContext null
            }
            LogUtil.i(
                TAG,
                "PART4B_ROUTING prefer_single=$preferSinglePart4b " +
                    "prefer_single_explicit=$preferSinglePart4bExplicit " +
                    "force_tiled_vulkan=$forceTiledPart4bOnVulkan " +
                    "has_single_pte=${part4bFile != null} has_alt_part4b=${hasPart4bNonSingleDecoder()} " +
                    "has_split_tileb2=${hasSplitTileB2Part4bVulkan()} " +
                    "has_split_tile00=${hasSplitTile00Part4bVulkan()} " +
                    "single_file=${part4bFile?.name ?: "none"} " +
                    "path=${
                        if (preferSinglePart4b) {
                            "single_decoder"
                        } else if (hasSplitTileB2Part4bVulkan()) {
                            "split_tile_b2_then_tile_b2_then_tile_b4_then_split_tile_00_then_tile_full"
                        } else if (hasSplitTile00Part4bVulkan()) {
                            "tile_b2_then_tile_b4_then_split_tile_00_then_tile_full"
                        } else {
                            "tile_b2_then_tile_b4_then_tile_00_then_tile_full"
                        }
                    } " +
                    "cpu_stable=$useCpuStable"
            )
            LogUtil.d(
                TAG,
                "Part4b: ${
                    if (preferSinglePart4b) {
                        "single"
                    } else if (hasSplitTileB2Part4bVulkan()) {
                        "tiled split-batched-first (tile_b2 stage_a/raw_heads on Vulkan, init_base/compose portable → legacy tile_b2 → tile_b4 → tile_00 split → tile_full → …)"
                    } else if (hasSplitTile00Part4bVulkan()) {
                        "tiled batched-first (tile_b2 batched → tile_b4 batched → tile_00 stage_a/raw_heads on Vulkan, init_base/compose portable → tile_full → …)"
                    } else {
                        "tiled batched-first (tile_b2 batched → tile_b4 batched → tile_00 sequential → tile_full → …)"
                    }
                } " +
                    "(C++ only, ${if (useCpuStable) "CPU" else "Vulkan"}${if (!useCpuStable && part12OnCpu) ", Part1+2 on CPU" else ""})"
            )
            if (!useCpuStable && part12OnCpu) {
                LogUtil.i(TAG, "[C++ FULL] Hybrid mode active: Part1+2 from models_cpu INT8, Part3/4 from models_vulkan")
            }
            Part1OnlyTest.suppressStartupWarmupForCurrentProcess("full_cpp_pipeline")

            val originalWidth = bitmap.width
            val originalHeight = bitmap.height
            val isPortrait = originalHeight > originalWidth
            val pipelineStartMs = System.currentTimeMillis()
            LogUtil.d(TAG, "[ASPECT] input=${originalWidth}x${originalHeight} | isPortrait=$isPortrait | ${if (USE_STRETCH_TO_SQUARE) "stretch-to-square" else "center-crop"} + raw coords (no aspect scale in PLY)")

            report(0f, "Preparing…", progressCallback)
            // 1. Stretch full image to 1536 (like Swift) or center-crop then resize; raw PLY coords
            val scaledBmp = if (USE_STRETCH_TO_SQUARE) getStretchToSquareBitmap(bitmap, IMAGE_SIZE) else getCenterCropBitmap(bitmap, IMAGE_SIZE)

            // Native-only: full SHARP graph in C++ (Part1–Part4b). Do not add a Kotlin inference path here.
            val minAvailForCpp = 400L * 1024 * 1024 // 400MB
            val tryCppFullPipeline = requiredSatisfied && NATIVE_FULL_AVAILABLE && availMem >= minAvailForCpp
            if (!tryCppFullPipeline) {
                if (!NATIVE_FULL_AVAILABLE) LogUtil.e(TAG, "[C++ FULL] Native full pipeline not available")
                else if (availMem < minAvailForCpp) LogUtil.d(TAG, "[C++ FULL] Low memory: avail=${availMem / (1024 * 1024)}MB")
                return@withContext null
            }
            val use1280Requested = prefs.getBoolean("executorch_int8_use_1280", false)
            val use1280 = false
            if (use1280Requested) {
                LogUtil.w(
                    TAG,
                    "[C++ FULL] executorch_int8_use_1280 requested, but the current hybrid split pipeline still " +
                        "depends on 1536-only Part4a/Part4b plus fixed low-res patch features. Ignoring the flag " +
                        "(1280 model set present=${hasTrue1280VulkanModelSet(preferVulkanFp16)})."
                )
            }
            // Same model dir as preload: Vulkan = Part1 vulkan dir; CPU = Part4b/tile dir
            val cppModelDir = if (useCpuStable) {
                val part4bOrTileFile = part4bFile
                    ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_B2)
                    ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_B4)
                    ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_00)
                    ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_FULL)
                part4bOrTileFile?.parent ?: modelsDir?.absolutePath ?: internalModelsDir.absolutePath
            } else {
                findVulkanPart1Pte(preferVulkanFp16)?.parent
                    ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_B2_STAGE_A_VULKAN)?.parent
                    ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_00_STAGE_A_VULKAN)?.parent
                    ?: findFile("sharp_split_part4b_vulkan.pte")?.parent
                    ?: modelsDir?.absolutePath ?: internalModelsDir.absolutePath
            }
            LogUtil.d(TAG, "[C++ FULL] modelDir=$cppModelDir useVulkan=${!useCpuStable} part12OnCpu=$part12OnCpu use1280=$use1280")
            run {
                val vk = !useCpuStable
                val p3 = if (vk) findVulkanPart3Pte(preferVulkanFp16)
                else {
                    findFile(SharpExecuTorchSplitModelNames.PART3_INT8)
                        ?: findFile(SharpExecuTorchSplitModelNames.PART3_FP16)
                        ?: findFile(SharpExecuTorchSplitModelNames.PART3_FP32)
                }
                val a512 = if (vk) findFile("sharp_split_part4a_chunk_512_vulkan.pte") else findFile(SharpExecuTorchSplitModelNames.PART4A_CHUNK_512)
                val a65 = if (vk) findFile("sharp_split_part4a_chunk_65_vulkan.pte") else findFile(SharpExecuTorchSplitModelNames.PART4A_CHUNK_65)
                val b4 = if (vk) findFile("sharp_split_part4b_vulkan.pte")
                else {
                    findSinglePart4bCpuPte()
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_B2)
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_B4)
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_00)
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_FULL)
                }
                LogUtil.i(
                    TAG,
                    "[C++ FULL] PTE (Kotlin findFile → absolute): P3=${p3?.absolutePath} P4a512=${a512?.absolutePath} " +
                        "P4a65=${a65?.absolutePath} P4b=${b4?.absolutePath} " +
                        "P4bSplitTileB2StageA=${findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_B2_STAGE_A_VULKAN)?.absolutePath} " +
                        "P4bSplitTile00StageA=${findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_00_STAGE_A_VULKAN)?.absolutePath} | cppModelDir=$cppModelDir"
                )
            }
            releasePreInferenceCaches("before_full_cpp_pipeline")
            val imageNCHW = if (use1280) {
                val bmp1280 = Bitmap.createScaledBitmap(scaledBmp, IMAGE_SIZE_1280, IMAGE_SIZE_1280, true)
                val bmp1536 = Bitmap.createScaledBitmap(bmp1280, IMAGE_SIZE, IMAGE_SIZE, true)
                bmp1280.recycle()
                preprocessToFloatArray(bmp1536, IMAGE_SIZE).also { bmp1536.recycle() }
            } else {
                preprocess(scaledBmp, false)
                imageFloatBuffer.rewind()
                FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE).also { imageFloatBuffer.get(it) }
            }
            if (!scaledBmp.isRecycled) scaledBmp.recycle()
            try {
                currentProgressCallback = progressCallback
                val cppStartMs = System.currentTimeMillis()
                val part12OnCpuRun = part12OnCpu
                // CPU ExecuTorch INT8: fixed — single-patch Part1+2, 25 patches only (skip 0.5x+0.25x).
                val part12ForceSingle = true
                val part12_25Only = true
                val part1Max1x = 25
                val part1Max05x = 0
                val part12Chunk1x = 0
                val part12Chunk05x = 0
                val part12YieldMs = 0
                val swapTileNdcXY = false
                val etdumpPath = getAndClearEtdumpOutputPath()
                val cppResult = safeJniFloatArrayResult {
                    runFullPipelineInt8Native(
                        cppModelDir,
                        imageNCHW,
                        effectiveMaxGaussians,
                        preferSinglePart4b,  // mirrors Settings Stable Part4b; native tries single on disk first
                        !useCpuStable,       // useVulkan (Part4 on Vulkan)
                        part12OnCpuRun,      // Part1+2 on CPU when true to avoid VK_ERROR_DEVICE_LOST
                        part12ForceSingle,   // Part1+2 single-patch only; avoids batch-4 SIGSEGV on some devices
                        part12_25Only,       // Skip 0.5x+0.25x to avoid SIGSEGV at patch 25/35 on some devices
                        part1Max1x,
                        part1Max05x,
                        part12Chunk1x,
                        part12Chunk05x,
                        part12YieldMs,
                        swapTileNdcXY,
                        this@ExecutorchInt8Sharp,  // progress reporter (reportProgressFromNative called from C++)
                        etdumpPath
                    )
                }
                val cppElapsedMs = System.currentTimeMillis() - cppStartMs
                if (cppResult != null && cppResult.isNotEmpty()) {
                    LogUtil.d(TAG, "[C++ FULL] ${cppResult.size / 14} Gaussians in ${cppElapsedMs}ms")
                    report(0.92f, "Saving your 3D room…", progressCallback)
                    val result = writePly(cppResult, progressCallback, isPortrait)
                    report(1f, "Your room is ready!", progressCallback)
                    return@withContext result
                }
                LogUtil.e(
                    TAG,
                    "[C++ FULL] Native returned null (no fallback). See native errors: " +
                        "adb logcat -d -s sharp_executorch_full:E sharp_executorch_full:W | tail -80"
                )
                return@withContext null
            } catch (e: Throwable) {
                LogUtil.e(TAG, "[C++ FULL] C++ failed: ${e.message}", e)
                return@withContext null
            } finally {
                currentProgressCallback = null
            }
        }
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

    /** Preprocess bitmap to NCHW float array [0,1] for a given size (e.g. 1280 for reduced memory). */
    private fun preprocessToFloatArray(bmp: Bitmap, size: Int): FloatArray {
        val total = size * size
        val out = FloatArray(3 * total)
        val pixels = IntArray(total)
        bmp.getPixels(pixels, 0, size, 0, 0, size, size)
        for (i in 0 until total) {
            val argb = pixels[i]
            out[i] = (argb shr 16 and 0xFF) / 255f
            out[total + i] = (argb shr 8 and 0xFF) / 255f
            out[2 * total + i] = (argb and 0xFF) / 255f
        }
        return out
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
