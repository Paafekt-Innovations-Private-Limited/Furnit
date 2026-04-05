package com.furnit.android.services

// BUILD_ID_TERMINAL_2025
import android.app.ActivityManager
import android.content.Context
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import com.furnit.android.BuildConfig
import com.furnit.android.ar.MetricAnchor
import com.furnit.android.utils.DebugLogger
import com.furnit.android.utils.DeviceHeuristics
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
 * Part1+2 use the 1× patch grid (default 16; settings allow 25 for full overlap) plus 0.25×; 0.5×/0.25× multi-scale may be skipped
 * in fixed modes. Vulkan/GPU work happens inside native
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

    /**
     * True while `runFullPipelineInt8Native` is executing. Used to ignore [releaseNativeCaches] from
     * [android.app.Application.onTrimMemory] / [android.content.ComponentCallbacks.onLowMemory], which would
     * otherwise free ModuleCache + workspace while native code is still using them (SIGSEGV).
     */
    @Volatile
    private var nativeFullPipelineRunning: Boolean = false

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

        /** Default when prefs/caller omit a cap: 0 = All (unlimited). */
        private const val DEFAULT_MAX_GAUSSIANS_ALL = 0

        /** Min free RAM (bytes) for native optional interleaved Vulkan Part1+2; see `docs/SHARP_HYBRID_OPTIMIZATION.md`. */
        private const val HYBRID_INTERLEAVE_MIN_AVAIL_BYTES = 512L * 1024 * 1024

        /**
         * Max 1× Part1 encoder patches (native `limit1x`). 25 = full 5×5 overlap grid; 16 = 4×4 (faster, softer edges).
         * Stored in [furnit_prefs] as an int; see [readPart1MaxPatches1xFromPrefs].
         */
        const val PREF_KEY_PART1_MAX_PATCHES_1X = "sharp_part1_max_patches_1x"
        private const val DEFAULT_PART1_MAX_PATCHES_1X = 16
        private const val MIN_PART1_MAX_PATCHES_1X = 16
        private const val MAX_PART1_MAX_PATCHES_1X = 25

        /**
         * Resolves to **16** (faster 4×4) or **25** (full 5×5) for C++ `part1MaxPatches1x`.
         * Any stored value other than 16 is treated as full quality (25).
         */
        fun readPart1MaxPatches1xFromPrefs(prefs: SharedPreferences): Int {
            val raw = prefs.getInt(PREF_KEY_PART1_MAX_PATCHES_1X, DEFAULT_PART1_MAX_PATCHES_1X)
            return if (raw == MIN_PART1_MAX_PATCHES_1X) MIN_PART1_MAX_PATCHES_1X else MAX_PART1_MAX_PATCHES_1X
        }

        private const val MODEL_FILENAME = "sharp_full_vulkan.pte"
        private val SPLIT_FILENAMES = SharpExecuTorchSplitModelNames.VULKAN_SPLIT_CORE_PTES
        /** Names of .pte files that may be packaged in assets (models_cpu / models_cpuvulkan_hybrid) for testing. */
        private val ASSET_MODEL_FILENAMES = SharpExecuTorchSplitModelNames.ASSET_OFFLOADABLE_VULKAN_PTES
        private const val PART4B_TILED_GRID = 4
        private const val PART4B_TILED_NUM = PART4B_TILED_GRID * PART4B_TILED_GRID // 16
        private const val PART4B_GAUSSIANS_PER_TILE = 73728
        /** When true, native C++ may use 2 modules in parallel (ignored in C++ for stability); read from prefs in inferStreaming. */
        @JvmField
        var PARALLEL_TILES: Boolean = false
        /** External + internal storage subdir per Gradle flavor: keeps CPU vs hybrid Vulkan .pte separate on device. */
        const val MODELS_SUBDIR_CPU = "models_cpu"
        /** etVulkan: CPU Part1+2 + Vulkan Part3/4 in one folder (internal + scoped external). */
        const val MODELS_SUBDIR_CPU_VULKAN_HYBRID = "models_cpuvulkan_hybrid"
        /** Legacy dir name; files are migrated into [MODELS_SUBDIR_CPU_VULKAN_HYBRID] on startup, then removed when empty. */
        private const val LEGACY_MODELS_SUBDIR_VULKAN = "models_vulkan"

        private const val LOGIT_LUT_SIZE = 1024
        private val LOGIT_LUT = FloatArray(LOGIT_LUT_SIZE) { i ->
            val p = (i.toFloat() / (LOGIT_LUT_SIZE - 1)).coerceIn(1e-4f, 1f - 1e-4f)
            ln(p / (1f - p))
        }

        private const val SRGB_LUT_SIZE = 4096
        private val SRGB_LUT = FloatArray(SRGB_LUT_SIZE) { i ->
            val v = i / (SRGB_LUT_SIZE - 1).toFloat()
            if (v <= LINEAR_TO_SRGB_THRESHOLD) {
                v * 12.92f
            } else {
                (1.055f * v.toDouble().pow(1.0 / 2.4).toFloat() - 0.055f).coerceIn(0f, 1f)
            }
        }

        private const val LN_LUT_SIZE = 2048
        private const val LN_LUT_MIN = 0.001f
        private const val LN_LUT_MAX = 5.0f

        /**
         * After copying external hybrid dir → internal, we optionally prune internal `sharp_split*.pte` that are
         * not listed on external (v2 adb push). **Never prune** when external has only a few stray files: that used to
         * wipe a full friend-APK bundle from internal storage and produced "Missing models" despite a multi‑GB APK.
         * Require at least this many `sharp_split*.pte` on external before pruning, unless external already has
         * at least as many files as internal did before sync (full replacement).
         */
        private const val MIN_EXTERNAL_SHARP_SPLIT_COUNT_FOR_STALE_PRUNE = 7
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

        @JvmStatic
        private external fun nativeSetSharpExecVerboseLogging(enabled: Boolean)

        @JvmStatic
        private external fun nativeSetSharpTilesVerboseLogging(enabled: Boolean)

        /**
         * Sync Settings debug_mode + debuggable APK to native `LOGD`/`LOGE` in `sharp_executorch_full` and
         * `sharp_executorch_tiles`. Call after changing the pref and before pipeline JNI.
         */
        fun syncSharpNativeVerboseLogging() {
            val on = DebugLogger.isSharpNativeVerboseEnabled
            try {
                if (NATIVE_FULL_AVAILABLE) nativeSetSharpExecVerboseLogging(on)
            } catch (_: Throwable) {
            }
            try {
                if (NATIVE_TILES_AVAILABLE) nativeSetSharpTilesVerboseLogging(on)
            } catch (_: Throwable) {
            }
        }

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
    /** Resolve flavor from manifest first so runtime follows the actually installed APK, not only generated BuildConfig. */
    private val usesVulkanAarRuntime: Boolean by lazy { resolveUsesVulkanAarRuntime() }
    /** Matches etVulkan vs etCpu APK: `files/models_cpuvulkan_hybrid` or `files/models_cpu` (+ scoped external storage). */
    private val executorchModelsSubdir: String
        get() = if (prefersHybridModelsDir()) MODELS_SUBDIR_CPU_VULKAN_HYBRID else MODELS_SUBDIR_CPU
    /** External dir for adb push: try type first, then base + subdir (some devices return null for type). */
    private val externalModelsDir: File?
        get() = context.getExternalFilesDir(executorchModelsSubdir)
            ?: context.getExternalFilesDir(null)?.let { File(it, executorchModelsSubdir).takeIf { d -> d.exists() || d.mkdirs() } }
    private val modelsDir: File
        get() = externalModelsDir ?: File(context.filesDir, executorchModelsSubdir)
    private val internalModelsDir: File
        get() = File(context.filesDir, executorchModelsSubdir).also { it.mkdirs() }

    /** Last user-visible failure from [inferStreaming]; consumed by [SharpService] for onError. */
    @Volatile
    private var inferStreamingFailureDetail: String? = null

    /** Set when [initialize] returns false (e.g. Vulkan hybrid without CPU Part1+2 sidecars). */
    @Volatile
    private var initializeFailureDetail: String? = null

    fun consumeInferStreamingFailureDetail(): String? {
        val message = inferStreamingFailureDetail
        inferStreamingFailureDetail = null
        return message
    }

    fun consumeInitializeFailureDetail(): String? {
        val message = initializeFailureDetail
        initializeFailureDetail = null
        return message
    }

    /** User-facing text when ExecuTorch Vulkan hybrid is selected but portable Part1+2 are missing. */
    private fun missingVulkanHybridPart12SidecarsUserMessage(): String {
        val n = SharpExecuTorchSplitModelNames
        return buildString {
            append("Vulkan hybrid requires CPU Part1+2 models: ")
            append("${n.PART1_INT8} and ${n.PART2_INT8} ")
            append("(or matching portable *_fp16 / *_fp32 pairs for both parts), ")
            append("in the same folder as your Vulkan Part3/4 stack (e.g. files/models_cpuvulkan_hybrid/). ")
            append("Or choose CPU ExecuTorch INT8 in Settings.")
        }
    }

    private fun resolveUsesVulkanAarRuntime(): Boolean {
        val fallback = BuildConfig.EXECUTORCH_USE_VULKAN_AAR
        return try {
            val applicationInfo = context.packageManager.getApplicationInfo(
                context.packageName,
                PackageManager.GET_META_DATA,
            )
            val metaValue = applicationInfo.metaData?.getBoolean("com.furnit.executorch.USE_VULKAN_AAR")
            metaValue ?: fallback
        } catch (_: Exception) {
            fallback
        }
    }

    private fun hasHybridModelsOnDisk(): Boolean {
        val hybridDirs = buildList {
            add(File(context.filesDir, MODELS_SUBDIR_CPU_VULKAN_HYBRID))
            context.getExternalFilesDir(MODELS_SUBDIR_CPU_VULKAN_HYBRID)?.let(::add)
            context.getExternalFilesDir(null)?.let { add(File(it, MODELS_SUBDIR_CPU_VULKAN_HYBRID)) }
        }.distinctBy { it.absolutePath }
        return hybridDirs.any { dir ->
            if (!dir.exists()) return@any false
            val entries = dir.listFiles().orEmpty()
            val hasPortablePart12 = entries.any { it.isFile && it.name == SharpExecuTorchSplitModelNames.PART1_INT8 } &&
                entries.any { it.isFile && it.name == SharpExecuTorchSplitModelNames.PART2_INT8 }
            val hasVulkanStack = entries.any { it.isFile && it.name.startsWith("sharp_split_part3_vulkan") } ||
                entries.any { it.isFile && it.name.startsWith("sharp_split_part4") }
            hasPortablePart12 && hasVulkanStack
        }
    }

    private fun prefersHybridModelsDir(): Boolean = usesVulkanAarRuntime || hasHybridModelsOnDisk()

    private val plyBatch = ByteBuffer.allocateDirect(BYTES_PER_VERTEX * PLY_BATCH_SIZE).apply { order(ByteOrder.LITTLE_ENDIAN) }
    private val zeroSHBytes = ByteArray(45 * 4)
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
            initializeFailureDetail = null
            // 1) Extract APK assets → files/models_cpu + files/models_cpuvulkan_hybrid (when Gradle bundled .pte).
            // 2) Copy adb-pushed files from scoped external storage into the same internal dirs.
            hydrateBundledAndExternalModels()
            if (NATIVE_FULL_AVAILABLE) {
                val prefs = context.getSharedPreferences("furnit_prefs", Context.MODE_PRIVATE)
                ExecutorchFixedSettings.syncToPrefs(prefs)
                val useCpuStable = prefs.getBoolean("executorch_int8_use_cpu_stable", false)
                val preferVulkanFp16Requested = ExecutorchFixedSettings.PREFER_VULKAN_FP16
                val preferVulkanFp16 = effectivePreferVulkanFp16(preferVulkanFp16Requested)
                if (preferVulkanFp16Requested && !preferVulkanFp16) {
                    LogUtil.w(
                        TAG,
                        "Ignoring 'Prefer Vulkan FP16 models' because this APK uses executorch-android-vulkan AAR " +
                            "and the required FP16 shaders are not bundled. Falling back to FP32-safe Vulkan models."
                    )
                }
                val hasP12Sidecars = hasCpuPart12SidecarModels()
                if (!useCpuStable && !hasP12Sidecars) {
                    val msg = missingVulkanHybridPart12SidecarsUserMessage()
                    initializeFailureDetail = msg
                    LogUtil.e(TAG, "[INIT] $msg")
                    return@withLock false
                }
                // Vulkan hybrid: CPU Part1+2 sidecars required (checked above). CPU-stable: portable Part1+2 under models_cpu.
                val effectivePart12OnCpu = !useCpuStable && hasP12Sidecars
                val cppModelDir = if (useCpuStable) {
                    findSinglePart4bCpuPte()?.parent
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_B2)?.parent
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_B4)?.parent
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_00)?.parent
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_FULL)?.parent
                        ?: findFile(SharpExecuTorchSplitModelNames.PART1_INT8)?.parent
                        ?: findFile(SharpExecuTorchSplitModelNames.PART1_FP32)?.parent
                } else {
                    // Same folder as Vulkan stack; sidecar Part1 path anchors dir when no Vulkan Part1 .pte exists.
                    findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART1_INT8)?.parent
                        ?: findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART1_FP32)?.parent
                        ?: findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART1_FP16)?.parent
                        ?: findVulkanPart3Pte(preferVulkanFp16)?.parent
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_B2)?.parent
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_B4)?.parent
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_00)?.parent
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_FULL)?.parent
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_B2_STAGE_A_VULKAN)?.parent
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_00_STAGE_PRE_VULKAN)?.parent
                } ?: modelsDir.absolutePath
                val useVulkanForPart12 = !useCpuStable && !effectivePart12OnCpu
                syncSharpNativeVerboseLogging()
                val preloaded = preloadCppModules(cppModelDir, useVulkanForPart12)
                LogUtil.d(TAG, "[C++ FULL] Preload Part1+Part2 cache: ${if (preloaded) "OK" else "failed"} (dir=$cppModelDir useVulkan=${!useCpuStable} part12OnCpu=$effectivePart12OnCpu)")
                if (!useCpuStable && effectivePart12OnCpu) {
                    LogUtil.i(TAG, "[C++ FULL] Hybrid active: Part1+2 INT8/portable (models_cpuvulkan_hybrid), Part3/4 Vulkan")
                }
                LogUtil.i(
                    TAG,
                    "EXECUTORCH_USE_VULKAN_AAR build=${BuildConfig.EXECUTORCH_USE_VULKAN_AAR} " +
                        "runtime=$usesVulkanAarRuntime (false = etCpu flavor / XNNPACK AAR; true = etVulkan). " +
                        "settings_cpu_stable=$useCpuStable"
                )
                val rootsHint =
                    if (prefersHybridModelsDir()) {
                        "etVulkan: push Vulkan Part3/4 + required portable Part1+2 (INT8 or fp16/fp32) to .../files/$MODELS_SUBDIR_CPU_VULKAN_HYBRID/"
                    } else {
                        "etCpu: push portable .pte to .../files/$MODELS_SUBDIR_CPU/"
                    }
                LogUtil.i(
                    TAG,
                    "ExecuTorch model roots: internal=${internalModelsDir.absolutePath} external=${modelsDir.absolutePath} ($rootsHint)",
                )
                if (!usesVulkanAarRuntime && !useCpuStable && !hasHybridModelsOnDisk()) {
                    LogUtil.w(
                        TAG,
                        "APK ExecuTorch is XNNPACK-only but Settings use Vulkan model layout — Vulkan .pte may fail. " +
                            "Either enable CPU ExecuTorch INT8 in Settings or install the etVulkan APK: ./gradlew :app:assembleEtVulkanDebug"
                    )
                }
            }
            isInitialized = true
            true
        }
    }

    private fun sharpSplitPteNamesInAssetSubdir(subdir: String): List<String> =
        try {
            context.assets.list(subdir)
                ?.asSequence()
                ?.filter { name ->
                    name.endsWith(".pte", ignoreCase = true) && name.startsWith("sharp_split")
                }
                ?.toList()
                .orEmpty()
        } catch (_: Exception) {
            emptyList()
        }

    /**
     * Copy `sharp_split*.pte` from one assets subdir into the matching internal dir (mmap-friendly).
     * Uses directory listing plus [ASSET_MODEL_FILENAMES] so optional names are still attempted.
     */
    private fun copySharpSplitPteFromAssetSubdirToInternal(assetSubdir: String, destInternalDir: File) {
        val filenamesFromAssets = sharpSplitPteNamesInAssetSubdir(assetSubdir)
        val filenamesToCopy = if (filenamesFromAssets.isNotEmpty()) {
            (filenamesFromAssets + ASSET_MODEL_FILENAMES.asList()).distinct()
        } else {
            ASSET_MODEL_FILENAMES.toList()
        }
        var tilesCopied = 0
        var tilesMissing = 0
        for (filename in filenamesToCopy) {
            // Vulkan .pte never live under assets/models_cpu; probing them there only spams logcat.
            if (assetSubdir == MODELS_SUBDIR_CPU && filename.contains("_vulkan", ignoreCase = true)) {
                continue
            }
            val dest = File(destInternalDir, filename)
            if (dest.exists() && dest.length() > 0L) {
                if (filename.startsWith("sharp_split_part4b_tile_")) tilesCopied++
                continue
            }
            val assetPath = "$assetSubdir/$filename"
            val isTile = filename.startsWith("sharp_split_part4b_tile_")
            try {
                context.assets.open(assetPath).use { input: InputStream ->
                    FileOutputStream(dest).use { output ->
                        input.copyTo(output)
                    }
                }
                if (isTile) tilesCopied++
                LogUtil.d(TAG, "Copied $filename from assets/$assetSubdir to ${dest.absolutePath}")
            } catch (_: Exception) {
                if (isTile) tilesMissing++
                // No per-file log: without bundled .pte (skipExecutorchAssets) or partial APK, misses are expected.
            }
        }
        if (tilesMissing > 0) {
            LogUtil.d(
                TAG,
                "Part4b tile models ($assetSubdir): ${tilesCopied}/17 from assets " +
                    "(add tile .pte to executorch_models or sharp_vulkan_only and rebuild for tiled path)",
            )
        } else if (tilesCopied == 17) {
            LogUtil.d(TAG, "Part4b tile models ready from assets/$assetSubdir (16 tiles + tile_full)")
        }
    }

    /**
     * Copy `sharp_split*.pte` from legacy `files/.../models_vulkan` into [MODELS_SUBDIR_CPU_VULKAN_HYBRID],
     * delete sources when the copy matches size, remove empty legacy dirs. Safe to run every launch.
     */
    private fun migrateLegacyModelsVulkanDirectoryToHybrid() {
        if (!prefersHybridModelsDir()) return
        val hybridName = MODELS_SUBDIR_CPU_VULKAN_HYBRID
        val pairs = LinkedHashSet<Pair<File, File>>()
        fun addPair(old: File, newDir: File) {
            if (old.absolutePath != newDir.absolutePath) {
                newDir.mkdirs()
                pairs.add(old to newDir)
            }
        }
        addPair(File(context.filesDir, LEGACY_MODELS_SUBDIR_VULKAN), File(context.filesDir, hybridName))
        context.getExternalFilesDir(LEGACY_MODELS_SUBDIR_VULKAN)?.let { oldExt ->
            val newExt =
                context.getExternalFilesDir(hybridName)
                    ?: context.getExternalFilesDir(null)?.let { root ->
                        File(root, hybridName).also { it.mkdirs() }
                    }
            if (newExt != null) addPair(oldExt, newExt)
        }
        context.getExternalFilesDir(null)?.let { base ->
            addPair(File(base, LEGACY_MODELS_SUBDIR_VULKAN), File(base, hybridName))
        }
        for ((oldDir, newDir) in pairs) {
            if (!oldDir.exists() || !oldDir.isDirectory) continue
            newDir.mkdirs()
            val pteFiles = oldDir.listFiles()?.filter { f ->
                f.isFile && f.name.startsWith("sharp_split", ignoreCase = true) && f.name.endsWith(".pte", ignoreCase = true)
            }.orEmpty()
            for (src in pteFiles) {
                val dst = File(newDir, src.name)
                try {
                    if (!dst.exists() || dst.length() == 0L) {
                        src.copyTo(dst, overwrite = true)
                        LogUtil.i(
                            TAG,
                            "[MODELS] Migrated ${src.name}: $LEGACY_MODELS_SUBDIR_VULKAN → $hybridName (${oldDir.absolutePath})",
                        )
                    }
                    if (dst.exists() && dst.length() > 0L && dst.length() == src.length()) {
                        if (src.delete()) {
                            LogUtil.d(TAG, "[MODELS] Removed legacy ${src.name} from ${oldDir.absolutePath}")
                        }
                    }
                } catch (e: Throwable) {
                    LogUtil.w(TAG, "[MODELS] Migrate ${src.name} failed: ${e.message}")
                }
            }
            try {
                val rest = oldDir.listFiles()
                if (rest.isNullOrEmpty()) {
                    if (oldDir.delete()) {
                        LogUtil.i(TAG, "[MODELS] Removed empty legacy directory ${oldDir.absolutePath}")
                    }
                }
            } catch (_: Throwable) { }
        }
    }

    /**
     * Copy packaged .pte from `assets/models_cpu` and `assets/models_cpuvulkan_hybrid` into internal storage
     * so mmap loads work. Hybrid APKs need both trees when bundled.
     */
    private fun ensureModelsFromAssets() {
        migrateLegacyModelsVulkanDirectoryToHybrid()
        val internalCpu = File(context.filesDir, MODELS_SUBDIR_CPU).also { it.mkdirs() }
        val internalVulkan = File(context.filesDir, MODELS_SUBDIR_CPU_VULKAN_HYBRID).also { it.mkdirs() }
        copySharpSplitPteFromAssetSubdirToInternal(MODELS_SUBDIR_CPU, internalCpu)
        copySharpSplitPteFromAssetSubdirToInternal(MODELS_SUBDIR_CPU_VULKAN_HYBRID, internalVulkan)
        syncExternalSharpSplitPteToInternal()
        syncExternalCpuSharpSplitPteToInternal()
    }

    /**
     * Copy bundled `sharp_split*.pte` from APK assets (`assets/models_cpu`, `assets/models_cpuvulkan_hybrid`) into app
     * internal storage, then overlay anything pushed to scoped external storage.
     * **Idempotent** — safe to call on every cold start / SHARP screen open. When `skipExecutorchAssets=true`
     * (default for fast local builds), APK assets have no .pte: use adb push, or build a friend APK with
     * `assemble_friend_apk_with_models.sh` / `-PskipExecutorchAssets=false`.
     */
    fun hydrateBundledAndExternalModels() {
        ensureModelsFromAssets()
        logInternalSplitPteCounts("hydrateBundledAndExternalModels")
    }

    private fun countSharpSplitPteInDir(dir: File): Int =
        dir.listFiles()?.count { f ->
            f.isFile && f.name.startsWith("sharp_split") && f.name.endsWith(".pte", ignoreCase = true)
        } ?: 0

    /**
     * True when neither internal `files/models_cpu` nor `files/models_cpuvulkan_hybrid` contains any `sharp_split*.pte`.
     * Used at app startup to decide whether to run [hydrateBundledAndExternalModels] (friend APK / first install).
     */
    fun internalSplitSharpModelsAbsent(): Boolean {
        fun countSplit(dir: File): Int =
            dir.listFiles()?.count { f ->
                f.isFile && f.name.startsWith("sharp_split") && f.name.endsWith(".pte", ignoreCase = true)
            } ?: 0
        val cpuDir = File(context.filesDir, MODELS_SUBDIR_CPU)
        val hybridDir = File(context.filesDir, MODELS_SUBDIR_CPU_VULKAN_HYBRID)
        return countSplit(cpuDir) == 0 && countSplit(hybridDir) == 0
    }

    private fun logInternalSplitPteCounts(reason: String) {
        fun countSplit(dir: File): Int =
            dir.listFiles()?.count { f ->
                f.isFile && f.name.startsWith("sharp_split") && f.name.endsWith(".pte", ignoreCase = true)
            } ?: 0
        val cpuDir = File(context.filesDir, MODELS_SUBDIR_CPU)
        val hybridDir = File(context.filesDir, MODELS_SUBDIR_CPU_VULKAN_HYBRID)
        val cpuN = countSplit(cpuDir)
        val hybridN = countSplit(hybridDir)
        LogUtil.i(
            TAG,
            "$reason: internal sharp_split*.pte count models_cpu=$cpuN models_cpuvulkan_hybrid=$hybridN " +
                "(if 0: enable bundling with skipExecutorchAssets=false or adb push to external files/)",
        )
    }

    /** Same as [hydrateBundledAndExternalModels] (kept for older call sites). */
    fun syncModelsFromExternal() {
        hydrateBundledAndExternalModels()
    }

    /**
     * Copy `sharp_split*.pte` from flavor-scoped external dir (adb push target) into internal storage
     * so mmap loads are fast. etCpu: .../files/models_cpu/ ; etVulkan: .../files/models_cpuvulkan_hybrid/ .
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
        val internalSplitCountBefore =
            internalModelsDir.listFiles()?.count { f ->
                f.isFile && f.name.startsWith("sharp_split") && f.name.endsWith(".pte", ignoreCase = true)
            } ?: 0
        val externalSplitCount = externalSharpSplitNames.size
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
        // After a full adb push, drop stale internal sharp_split*.pte (e.g. old part4b.pte). Do not prune when
        // external has fewer files than a plausible full stack and fewer than we already had internally — that
        // indicates leftover/partial files, not an intentional replacement (friend APK hydrate would be wiped).
        val shouldPruneStaleInternal =
            externalSharpSplitNames.isNotEmpty() &&
                (
                    externalSplitCount >= internalSplitCountBefore ||
                        (
                            externalSplitCount >= MIN_EXTERNAL_SHARP_SPLIT_COUNT_FOR_STALE_PRUNE &&
                                externalSplitCount < internalSplitCountBefore
                            )
                    )
        var anyPruned = false
        if (shouldPruneStaleInternal) {
            internalModelsDir.listFiles()?.forEach { internalFile ->
                if (!internalFile.isFile || !internalFile.name.endsWith(".pte", ignoreCase = true)) return@forEach
                if (!internalFile.name.startsWith("sharp_split")) return@forEach
                if (externalSharpSplitNames.contains(internalFile.name)) return@forEach
                try {
                    if (internalFile.delete()) {
                        LogUtil.i(TAG, "Pruned internal ${executorchModelsSubdir} (not on external): ${internalFile.name}")
                        anyPruned = true
                    }
                } catch (e: Exception) {
                    LogUtil.w(TAG, "Could not prune ${internalFile.name}: ${e.message}")
                }
            }
        } else if (externalSharpSplitNames.isNotEmpty()) {
            LogUtil.w(
                TAG,
                "sync: skip internal prune (partial external push: external sharp_split*.pte=$externalSplitCount, " +
                    "internal before sync=$internalSplitCountBefore). Clear $externalDir or push a full model set to replace bundled files.",
            )
        } else {
            LogUtil.d(TAG, "sync: no sharp_split*.pte on external — skip internal prune")
        }
        if ((anyCopied || anyPruned) && NATIVE_FULL_AVAILABLE) {
            try {
                releaseCppModulesIfSafe()
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
                releaseCppModulesIfSafe()
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
     * Resolve a model file under flavor dirs only: [internalModelsDir] then [modelsDir] (`models_cpu` or hybrid dir).
     * Use scoped external `files/models_cpu` / `files/models_cpuvulkan_hybrid` (adb push); legacy `files/models` is not used.
     */
    private fun findFile(filename: String): File? {
        File(internalModelsDir, filename).takeIf { it.exists() && it.length() > 0L }?.let { return it }
        File(modelsDir, filename).takeIf { it.exists() && it.length() > 0L }?.let { return it }
        return null
    }

    /**
     * Part1+2 portable/INT8 weights for hybrid Vulkan runs.
     * **Prefer `models_cpuvulkan_hybrid`** (same folder as Vulkan Part3/4 + adb `push_sharp_cpuvulkan_hybrid_androidstudio.sh`), then `models_cpu` (legacy).
     */
    private fun findCpuSidecarFile(filename: String): File? {
        val internalVk = File(context.filesDir, MODELS_SUBDIR_CPU_VULKAN_HYBRID)
        File(internalVk, filename).takeIf { it.exists() && it.length() > 0L }?.let { return it }
        context.getExternalFilesDir(MODELS_SUBDIR_CPU_VULKAN_HYBRID)?.let { dir ->
            File(dir, filename).takeIf { it.exists() && it.length() > 0L }?.let { return it }
        }
        context.getExternalFilesDir(null)?.let { root ->
            File(root, MODELS_SUBDIR_CPU_VULKAN_HYBRID).let { dir ->
                File(dir, filename).takeIf { it.exists() && it.length() > 0L }?.let { return it }
            }
        }
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
        requested && !usesVulkanAarRuntime

    /** Default order is FP32-first; low-memory Vulkan mode flips this to FP16-first when such exports exist. */
    private fun findVulkanPart1Pte(preferFp16: Boolean = false): File? =
        orderedVulkanPrecisionNames(SharpExecuTorchSplitModelNames.VULKAN_RESOLVE_STEM_PART1, preferFp16).firstNotNullOfOrNull(::findFile)

    private fun findVulkanPart2Pte(preferFp16: Boolean = false): File? =
        orderedVulkanPrecisionNames(SharpExecuTorchSplitModelNames.VULKAN_RESOLVE_STEM_PART2, preferFp16).firstNotNullOfOrNull(::findFile)

    private fun findVulkanPart3Pte(preferFp16: Boolean = false): File? =
        orderedVulkanPrecisionNames("sharp_split_part3", preferFp16).firstNotNullOfOrNull(::findFile)

    private fun findVulkan1280Part3Pte(preferFp16: Boolean = false): File? =
        orderedVulkanPrecisionNames(SharpExecuTorchSplitModelNames.VULKAN_RESOLVE_STEM_PART3_1280, preferFp16).firstNotNullOfOrNull(::findFile)

    private fun findVulkan1280Part4Pte(preferFp16: Boolean = false): File? =
        orderedVulkanPrecisionNames(SharpExecuTorchSplitModelNames.VULKAN_RESOLVE_STEM_PART4_1280, preferFp16).firstNotNullOfOrNull(::findFile)

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
     * @param preferSinglePart4b CPU-stable only: when true, native may try single Part4b before tiled. Vulkan ignores (tiled-first).
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
        etdumpOutputPath: String?,
        hybridInterleavePart12: Boolean,
        hybridInterleaveMinAvailMemBytes: Long,
    ): FloatArray?

    private external fun getLastMonodepthInfoNative(): IntArray?

    private external fun getLastMonodepthBufferNative(): FloatArray?

    private external fun sampleMonodepthAtPointsNative(pixelXs: IntArray, pixelYs: IntArray, channel: Int): FloatArray?

    private external fun writePlyNative(
        outputPath: String,
        params: FloatArray,
        aspectCorrX: Float,
        aspectCorrY: Float,
        metricScale: Float,
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
                hasInvalidFloatSample(result) -> {
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

    private fun hasInvalidFloatSample(values: FloatArray, maxSamples: Int = 1024): Boolean {
        if (values.isEmpty()) return false
        val step = max(1, values.size / maxSamples)
        var index = 0
        while (index < values.size) {
            val value = values[index]
            if (!value.isFinite()) return true
            index += step
        }
        val last = values[values.lastIndex]
        return !last.isFinite()
    }

    private fun getLastMonodepthInfo(): MetricScaleEstimator.SharpMonodepthInfo? {
        val info = getLastMonodepthInfoNative() ?: return null
        if (info.size < 3) return null
        val width = info[0]
        val height = info[1]
        val channels = info[2]
        if (width <= 0 || height <= 0 || channels <= 0) return null
        return MetricScaleEstimator.SharpMonodepthInfo(
            width = width,
            height = height,
            channels = channels,
        )
    }

    /**
     * Persists the last pipeline monodepth to [folder]/sharp_monodepth.bin for offline YOLO wall measurement.
     * Header: 12 bytes little-endian (width, height, channels as int32) + float32 samples (row-major, c interleaved if c>1).
     * Call immediately after a successful [inferStreaming] while native buffers are still valid.
     */
    fun persistLastMonodepthToFolder(folder: File): Boolean {
        val info = getLastMonodepthInfoNative() ?: return false
        if (info.size < 3) return false
        val w = info[0]
        val h = info[1]
        val c = info[2]
        if (w <= 0 || h <= 0 || c <= 0) return false
        val buf = getLastMonodepthBufferNative() ?: return false
        val expected = w * h * c
        if (buf.size != expected) {
            LogUtil.w(TAG, "persistLastMonodepthToFolder: size mismatch got=${buf.size} expected=$expected (${w}x${h}x$c)")
            return false
        }
        return try {
            folder.mkdirs()
            val outFile = File(folder, "sharp_monodepth.bin")
            FileOutputStream(outFile).use { fos ->
                val header = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN)
                header.putInt(w)
                header.putInt(h)
                header.putInt(c)
                fos.write(header.array())
                val bb = ByteBuffer.allocate(buf.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                for (f in buf) {
                    bb.putFloat(f)
                }
                fos.write(bb.array())
            }
            LogUtil.d(TAG, "persistLastMonodepthToFolder: wrote ${outFile.absolutePath} ($w×$h×${c})")
            LogUtil.i("WALL_MEAS", "persist_monodepth ok path=${outFile.absolutePath} size=${w}x${h}x$c")
            true
        } catch (e: Exception) {
            LogUtil.e(TAG, "persistLastMonodepthToFolder failed: ${e.message}", e)
            false
        }
    }

    private fun sampleMonodepthChannel(xs: IntArray, ys: IntArray, channel: Int): FloatArray? {
        if (xs.size != ys.size) {
            LogUtil.w(TAG, "Native monodepth sample size mismatch: xs=${xs.size} ys=${ys.size}")
            return null
        }
        return try {
            val values = sampleMonodepthAtPointsNative(xs, ys, channel) ?: return null
            if (values.size != xs.size) {
                LogUtil.w(TAG, "Native monodepth sample result mismatch: got=${values.size} expected=${xs.size} channel=$channel")
                return null
            }
            values
        } catch (t: Throwable) {
            LogUtil.e(TAG, "Native monodepth sample failed: ${t.message}", t)
            null
        }
    }

    private fun tryWritePlyNative(
        outputPath: String,
        params: FloatArray,
        aspectCorrX: Float,
        aspectCorrY: Float,
        metricScale: Float,
    ): FloatArray? {
        return try {
            val stats = writePlyNative(outputPath, params, aspectCorrX, aspectCorrY, metricScale) ?: return null
            if (stats.size != 9 || hasInvalidFloatSample(stats, maxSamples = 9)) {
                LogUtil.w(TAG, "Native PLY stats invalid: size=${stats.size}")
                return null
            }
            stats
        } catch (t: Throwable) {
            LogUtil.e(TAG, "Native PLY export failed: ${t.message}", t)
            null
        }
    }

    private fun estimateGaussianDepthProxyUnits(params: FloatArray): Float {
        val depthCandidates = FloatArray(params.size / PARAMS_PER_GAUSSIAN)
        var count = 0
        for (offset in params.indices step PARAMS_PER_GAUSSIAN) {
            val opacity = params[offset + 3]
            if (!opacity.isFinite() || opacity < 0.15f) continue
            val depth = abs(params[offset + 2])
            if (!depth.isFinite() || depth <= 0.01f) continue
            depthCandidates[count] = depth
            count++
        }
        if (count == 0) return Float.NaN
        Arrays.sort(depthCandidates, 0, count)
        return depthCandidates[count / 2]
    }

    /** Warm-start: preload Part1+Part2. Vulkan path = Vulkan only (no CPU fallback). */
    private external fun preloadCppModules(modelDirPath: String, useVulkanForPart12: Boolean): Boolean

    /** Release native singleton Part1+Part2 cache and aligned workspace buffers. */
    private external fun releaseCppModules()

    private fun releaseCppModulesIfSafe() {
        if (nativeFullPipelineRunning) {
            LogUtil.w(TAG, "Skipping releaseCppModules — native full SHARP pipeline is running")
            return
        }
        releaseCppModules()
    }

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
        if (nativeFullPipelineRunning) {
            LogUtil.w(TAG, "Skipping releaseNativeCaches — native full SHARP pipeline is running (e.g. onTrimMemory)")
            return
        }
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

    @Suppress("UNUSED_PARAMETER")
    suspend fun inferStreaming(
        bitmap: Bitmap,
        metricAnchors: List<MetricAnchor>? = null,
        progressCallback: ((Float, String) -> Unit)? = null,
        useVulkan: Boolean = true,
        maxGaussians: Int = DEFAULT_MAX_GAUSSIANS_ALL,
    ): StreamingResult? = withContext(Dispatchers.IO) {
        mutex.withLock {
            inferStreamingFailureDetail = null
            if (!isInitialized) {
                inferStreamingFailureDetail = "SHARP is not initialized."
                return@withContext null
            }
            LogUtil.d(TAG, "[MEMORY] ${getMemoryInfo()}")
            ensureModelsFromAssets()

            val prefs = context.getSharedPreferences("furnit_prefs", Context.MODE_PRIVATE)
            ExecutorchFixedSettings.syncToPrefs(prefs)
            val useCpuStable = prefs.getBoolean("executorch_int8_use_cpu_stable", false)
            val preferVulkanFp16Requested = ExecutorchFixedSettings.PREFER_VULKAN_FP16
            val preferVulkanFp16 = effectivePreferVulkanFp16(preferVulkanFp16Requested)
            if (preferVulkanFp16Requested && !preferVulkanFp16) {
                LogUtil.w(
                    TAG,
                    "Ignoring 'Prefer Vulkan FP16 models' because this APK uses executorch-android-vulkan AAR " +
                        "and the required FP16 shaders are not bundled. Falling back to FP32-safe Vulkan models."
                )
            }
            val hasP12SidecarsInfer = hasCpuPart12SidecarModels()
            if (!useCpuStable && !hasP12SidecarsInfer) {
                val msg = missingVulkanHybridPart12SidecarsUserMessage()
                inferStreamingFailureDetail = msg
                LogUtil.e(TAG, "[C++ FULL] $msg")
                return@withContext null
            }
            val part12OnCpu = !useCpuStable && hasP12SidecarsInfer
            // Vulkan hybrid: Part1+2 portable sidecars only (no Vulkan Part1+2 fallback). CPU-stable: models_cpu.
            fun hasPart1() = if (useCpuStable) {
                findFile(SharpExecuTorchSplitModelNames.PART1_INT8) != null ||
                    findFile(SharpExecuTorchSplitModelNames.PART1_FP32) != null ||
                    findFile(SharpExecuTorchSplitModelNames.PART1_FP16) != null
            } else {
                findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART1_INT8) != null ||
                    findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART1_FP32) != null ||
                    findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART1_FP16) != null
            }
            fun hasPart2() = if (useCpuStable) {
                findFile(SharpExecuTorchSplitModelNames.PART2_INT8) != null ||
                    findFile(SharpExecuTorchSplitModelNames.PART2_FP32) != null ||
                    findFile(SharpExecuTorchSplitModelNames.PART2_FP16) != null
            } else {
                findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART2_INT8) != null ||
                    findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART2_FP32) != null ||
                    findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART2_FP16) != null
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
            else findFile(SharpExecuTorchSplitModelNames.PART4A_CHUNK_512_VULKAN) != null
            fun hasPart4a65() = if (allCpu) findFile(SharpExecuTorchSplitModelNames.PART4A_CHUNK_65) != null
            else findFile(SharpExecuTorchSplitModelNames.PART4A_CHUNK_65_VULKAN) != null
            // Vulkan / hybrid: Part4b is tiled-only in native (no sharp_split_part4b_vulkan.pte).
            val part4bFile = if (useCpuStable) findSinglePart4bCpuPte() else null
            /** Matches C++ `runPart4bTiledFullPipeline` / batched: fine-split tile_00 (stage_pre + decoder_head + shared stack). */
            fun hasFineSplitTile00Part4bVulkan(): Boolean {
                if (useCpuStable) return false
                val n = SharpExecuTorchSplitModelNames
                return findFile(n.PART4B_TILE_00_STAGE_PRE_VULKAN) != null &&
                    findFile(n.PART4B_TILE_00_DECODER_HEAD) != null &&
                    findFile(n.PART4B_TILE_00_INIT_BASE) != null &&
                    findFile(n.PART4B_TILE_00_RAW_HEADS_VULKAN) != null &&
                    findFile(n.PART4B_TILE_00_COMPOSE) != null
            }
            fun hasSplitTile00Part4bVulkan(): Boolean {
                if (useCpuStable) return false
                val n = SharpExecuTorchSplitModelNames
                return findFile(n.PART4B_TILE_00_STAGE_A_VULKAN) != null &&
                    findFile(n.PART4B_TILE_00_INIT_BASE) != null &&
                    findFile(n.PART4B_TILE_00_RAW_HEADS_VULKAN) != null &&
                    findFile(n.PART4B_TILE_00_COMPOSE) != null
            }
            /** Matches C++ batched path: fine-split tile_b2 (no stage_a). */
            fun hasFineSplitTileB2Part4bVulkan(): Boolean {
                if (useCpuStable) return false
                val n = SharpExecuTorchSplitModelNames
                return findFile(n.PART4B_TILE_B2_STAGE_PRE_VULKAN) != null &&
                    findFile(n.PART4B_TILE_B2_DECODER_HEAD) != null &&
                    findFile(n.PART4B_TILE_B2_INIT_BASE) != null &&
                    findFile(n.PART4B_TILE_B2_RAW_HEADS_VULKAN) != null &&
                    findFile(n.PART4B_TILE_B2_COMPOSE) != null
            }
            fun hasSplitTileB2Part4bVulkan(): Boolean {
                if (useCpuStable) return false
                val n = SharpExecuTorchSplitModelNames
                return findFile(n.PART4B_TILE_B2_STAGE_A_VULKAN) != null &&
                    findFile(n.PART4B_TILE_B2_INIT_BASE) != null &&
                    findFile(n.PART4B_TILE_B2_RAW_HEADS_VULKAN) != null &&
                    findFile(n.PART4B_TILE_B2_COMPOSE) != null
            }
            // Any one complete strategy matches native (sharp_executorch_full_common.cpp): see README models_cpuvulkan_hybrid.
            // Vulkan C++ order: batched tiled → sequential tiled (fine tile_00 first); no single part4b_vulkan .pte.
            fun hasPart4bNonSingleDecoder(): Boolean {
                val n = SharpExecuTorchSplitModelNames
                return hasFineSplitTile00Part4bVulkan() ||
                    hasFineSplitTileB2Part4bVulkan() ||
                    hasSplitTileB2Part4bVulkan() ||
                    hasSplitTile00Part4bVulkan() ||
                    findFile(n.PART4B_TILE_B2) != null ||
                    findFile(n.PART4B_TILE_B4) != null ||
                    findFile(n.PART4B_TILE_00) != null ||
                    findFile(n.PART4B_TILE_FULL) != null
            }
            val hasPart4bTiled = hasPart4bNonSingleDecoder()
            val part4bSatisfied = part4bFile != null || hasPart4bTiled
            // Vulkan / hybrid Vulkan: Part4b is tiled-only in native (batched → sequential).
            // CPU-stable full pipeline still passes preferSinglePart4b=true so C++ may try single decoder first.
            val preferSinglePart4b = useCpuStable
            // Effective Gaussian cap for full C++ pipeline. 0 = All (unlimited, no pruning).
            var effectiveMaxGaussians = prefs.getInt("executorch_int8_max_gaussians", DEFAULT_MAX_GAUSSIANS_ALL)
            val availMem = getAvailMemBytes()
            if (availMem < 600L * 1024 * 1024 && (effectiveMaxGaussians <= 0 || effectiveMaxGaussians > 300_000)) {
                val capped = 300_000
                LogUtil.d(TAG, "[MEMORY] Capping maxGaussians to $capped (avail=${availMem / (1024 * 1024)}MB)")
                effectiveMaxGaussians = capped
            }
            val requiredSatisfied = hasPart1() && hasPart2() && hasPart3() && hasPart4a512() && hasPart4a65() && part4bSatisfied
            if (!requiredSatisfied) {
                val missing = mutableListOf<String>()
                if (!hasPart1()) {
                    missing.add(
                        if (useCpuStable) "part1 portable" else "part1 portable sidecar (int8/fp16/fp32) for hybrid",
                    )
                }
                if (!hasPart2()) {
                    missing.add(
                        if (useCpuStable) "part2 portable" else "part2 portable sidecar (int8/fp16/fp32) for hybrid",
                    )
                }
                if (!hasPart3()) missing.add(if (useCpuStable) "part3 portable" else "part3 vulkan")
                if (!hasPart4a512()) missing.add(if (useCpuStable) "part4a_512" else "part4a_512_vulkan")
                if (!hasPart4a65()) missing.add(if (useCpuStable) "part4a_65" else "part4a_65_vulkan")
                if (!part4bSatisfied) {
                    missing.add(
                        if (useCpuStable) {
                            "part4b_int8 / part4b_fp16 / part4b.pte / part4b_tile_b2 / part4b_tile_b4 / tile_00 / tile_full"
                        } else {
                            "complete Part4b set: fine-split tile_00 (5), split tile_00 (4), fine/split tile_b2 (5/4), legacy tile_b2.pte/tile_b4.pte, or tile_00.pte/tile_full.pte"
                        }
                    )
                }
                val dirHint = modelsDir.absolutePath
                val hybridHint = if (!useCpuStable && part12OnCpu) {
                    "$dirHint (put INT8 Part1+2 and Vulkan Part3/4 in this models_cpuvulkan_hybrid folder; models_cpu optional)"
                } else {
                    dirHint
                }
                LogUtil.e(
                    TAG,
                    "ExecuTorch INT8 models missing (${if (useCpuStable) "CPU" else "Vulkan"}): $missing. Push to $hybridHint",
                )
                inferStreamingFailureDetail = buildString {
                    append("Missing models: ")
                    append(missing.joinToString(", "))
                    append(". ")
                    if (!useCpuStable && part12OnCpu) {
                        append("Hybrid: push all .pte to models_cpuvulkan_hybrid (INT8 Part1+2 + Vulkan stack). ")
                    }
                    append("Path: ")
                    append(hybridHint)
                }
                return@withContext null
            }
            LogUtil.i(
                TAG,
                "PART4B_ROUTING prefer_single=$preferSinglePart4b " +
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
                LogUtil.i(TAG, "[C++ FULL] Hybrid mode active: Part1+2 INT8/portable (models_cpuvulkan_hybrid preferred), Part3/4 Vulkan")
            }
            Part1OnlyTest.suppressStartupWarmupForCurrentProcess("full_cpp_pipeline")

            val originalWidth = bitmap.width
            val originalHeight = bitmap.height
            val isPortrait = originalHeight > originalWidth
            LogUtil.d(TAG, "[ASPECT] input=${originalWidth}x${originalHeight} | isPortrait=$isPortrait | ${if (USE_STRETCH_TO_SQUARE) "stretch-to-square" else "center-crop"} + raw coords (no aspect scale in PLY)")

            report(0f, "Preparing…", progressCallback)
            // 1. Stretch full image to 1536 (like Swift) or center-crop then resize; raw PLY coords
            val scaledBmp = if (USE_STRETCH_TO_SQUARE) getStretchToSquareBitmap(bitmap, IMAGE_SIZE) else getCenterCropBitmap(bitmap, IMAGE_SIZE)

            // Native-only: full SHARP graph in C++ (Part1–Part4b). Do not add a Kotlin inference path here.
            val minAvailForCpp =
                if (DeviceHeuristics.isGooglePixelFamily()) 280L * 1024 * 1024 else 400L * 1024 * 1024
            val tryCppFullPipeline = requiredSatisfied && NATIVE_FULL_AVAILABLE && availMem >= minAvailForCpp
            if (!tryCppFullPipeline) {
                if (!NATIVE_FULL_AVAILABLE) {
                    LogUtil.e(TAG, "[C++ FULL] Native full pipeline not available")
                    inferStreamingFailureDetail = "C++ SHARP library not loaded. Use an etVulkan (or etCpu) build with native SHARP enabled."
                } else if (availMem < minAvailForCpp) {
                    LogUtil.d(TAG, "[C++ FULL] Low memory: avail=${availMem / (1024 * 1024)}MB (need ~${minAvailForCpp / (1024 * 1024)}MB)")
                    inferStreamingFailureDetail =
                        "Low free RAM (${availMem / (1024 * 1024)} MB). Close other apps and try again."
                } else {
                    inferStreamingFailureDetail = "Cannot start the native SHARP pipeline."
                }
                return@withContext null
            }
            val use1280Requested = ExecutorchFixedSettings.USE_TRUE_1280
            val use1280 = ExecutorchFixedSettings.USE_TRUE_1280
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
                part4bOrTileFile?.parent ?: modelsDir.absolutePath
            } else {
                findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART1_INT8)?.parent
                    ?: findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART1_FP32)?.parent
                    ?: findCpuSidecarFile(SharpExecuTorchSplitModelNames.PART1_FP16)?.parent
                    ?: findVulkanPart3Pte(preferVulkanFp16)?.parent
                    ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_B2)?.parent
                    ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_B4)?.parent
                    ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_00)?.parent
                    ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_FULL)?.parent
                    ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_B2_STAGE_A_VULKAN)?.parent
                    ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_00_STAGE_A_VULKAN)?.parent
                    ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_00_STAGE_PRE_VULKAN)?.parent
                    ?: modelsDir.absolutePath
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
                val a512 = if (vk) findFile(SharpExecuTorchSplitModelNames.PART4A_CHUNK_512_VULKAN) else findFile(SharpExecuTorchSplitModelNames.PART4A_CHUNK_512)
                val a65 = if (vk) findFile(SharpExecuTorchSplitModelNames.PART4A_CHUNK_65_VULKAN) else findFile(SharpExecuTorchSplitModelNames.PART4A_CHUNK_65)
                val b4 = if (vk) {
                    findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_B2)
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_B4)
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_00)
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_FULL)
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_B2_STAGE_A_VULKAN)
                        ?: findFile(SharpExecuTorchSplitModelNames.PART4B_TILE_00_STAGE_PRE_VULKAN)
                } else {
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
                // CPU ExecuTorch INT8: fixed — single-patch Part1+2 (skip 0.5x+0.25x when part12_25Only).
                val part12ForceSingle = true
                val part12_25Only = true
                val part1Max1x = readPart1MaxPatches1xFromPrefs(prefs)
                if (part1Max1x < MAX_PART1_MAX_PATCHES_1X) {
                    LogUtil.d(
                        TAG,
                        "[C++ FULL] Part1 1× patch cap=$part1Max1x (faster than full 25; patch edges may be softer)",
                    )
                }
                val part1Max05x = 0
                val part12Chunk1x = 0
                val part12Chunk05x = 0
                val part12YieldMs = 0
                val swapTileNdcXY = false
                val etdumpPath = getAndClearEtdumpOutputPath()
                syncSharpNativeVerboseLogging()
                val hybridInterleaveRequested = prefs.getBoolean("sharp_hybrid_interleave_part12", false)
                val availMemBytes = getAvailMemBytes()
                val hybridInterleavePart12 =
                    hybridInterleaveRequested && availMemBytes >= HYBRID_INTERLEAVE_MIN_AVAIL_BYTES
                if (hybridInterleaveRequested && !hybridInterleavePart12) {
                    LogUtil.d(
                        TAG,
                        "[C++ FULL] [HYBRID] interleave Part1+2 skipped: avail=${availMemBytes / (1024 * 1024)}MB " +
                            "need >=${HYBRID_INTERLEAVE_MIN_AVAIL_BYTES / (1024 * 1024)}MB",
                    )
                }
                LogUtil.d(
                    TAG,
                    "[C++ FULL] [HYBRID] interleavePart12=$hybridInterleavePart12 " +
                        "interleaveGateBytes=$HYBRID_INTERLEAVE_MIN_AVAIL_BYTES",
                )
                nativeFullPipelineRunning = true
                val cppResult = try {
                    safeJniFloatArrayResult {
                        runFullPipelineInt8Native(
                            cppModelDir,
                            imageNCHW,
                            effectiveMaxGaussians,
                            preferSinglePart4b,  // true only for CPU-stable; Vulkan always tiled-first in native
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
                            etdumpPath,
                            hybridInterleavePart12,
                            HYBRID_INTERLEAVE_MIN_AVAIL_BYTES,
                        )
                    }
                } finally {
                    nativeFullPipelineRunning = false
                }
                val cppElapsedMs = System.currentTimeMillis() - cppStartMs
                if (cppResult != null && cppResult.isNotEmpty()) {
                    LogUtil.d(TAG, "[C++ FULL] ${cppResult.size / 14} Gaussians in ${cppElapsedMs}ms")
                    // Native Part4b ends ~0.90; keep PLY phase 0.91–1.0 so UI does not jump after long decoder.
                    // Wires PLY to preview folder only — "Save" in the app means Save from the viewer (library).
                    report(0.91f, "Writing your room file…", progressCallback)
                    val result = writePly(cppResult, progressCallback, isPortrait, originalWidth, originalHeight, metricAnchors)
                    report(1f, "Your room is ready!", progressCallback)
                    return@withContext result
                }
                inferStreamingFailureDetail = buildString {
                    append("SHARP GPU pipeline failed. ")
                    append("Log: adb logcat -d -s sharp_executorch_full:E ExecutorchInt8Sharp:E | tail -80")
                }
                LogUtil.e(TAG, "[C++ FULL] Native returned null (no fallback). $inferStreamingFailureDetail")
                return@withContext null
            } catch (e: Throwable) {
                LogUtil.e(TAG, "[C++ FULL] C++ failed: ${e.message}", e)
                inferStreamingFailureDetail =
                    (e.message ?: e.toString()).take(500) + " — see logcat sharp_executorch_full / ExecutorchInt8Sharp"
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
        return SRGB_LUT[(v * (SRGB_LUT_SIZE - 1)).toInt().coerceIn(0, SRGB_LUT_SIZE - 1)]
    }

    /**
     * Write PLY from model output. Uses raw coordinates (no aspect scaling): model trained on square input
     * expects 1:1 coordinate space. Only y,z are negated for our viewer convention.
     * See Ultralytics: apply aspectRatio only when mapping to non-square frustum; else normalization issues cause jagged output.
     */
    private fun writePly(
        params: FloatArray,
        progressCallback: ((Float, String) -> Unit)?,
        isPortrait: Boolean = false,
        originalImageWidth: Int = 0,
        originalImageHeight: Int = 0,
        metricAnchors: List<MetricAnchor>? = null,
    ): StreamingResult {
        val count = params.size / PARAMS_PER_GAUSSIAN

        val imgW = if (originalImageWidth > 0) originalImageWidth else IMAGE_SIZE
        val imgH = if (originalImageHeight > 0) originalImageHeight else IMAGE_SIZE

        // Aspect-ratio correction for stretch-to-square.
        // The model treats the 1536×1536 input as isotropic, producing a square bounding box.
        // To recover the original image's aspect ratio, scale each axis by (S/dim) (inverse
        // of the stretch), then normalize so the geometric mean is 1 (preserving overall scale).
        val aspectCorrX: Float
        val aspectCorrY: Float
        if (USE_STRETCH_TO_SQUARE && imgW != imgH) {
            val rawCorrX = IMAGE_SIZE.toFloat() / imgW.toFloat()
            val rawCorrY = IMAGE_SIZE.toFloat() / imgH.toFloat()
            val geomMean = sqrt(rawCorrX * rawCorrY)
            aspectCorrX = rawCorrX / geomMean
            aspectCorrY = rawCorrY / geomMean
        } else {
            aspectCorrX = 1f
            aspectCorrY = 1f
        }
        LogUtil.d(TAG, "[PLY] writePly: count=$count isPortrait=$isPortrait imgDims=${imgW}x${imgH} aspectCorr=(${"%.4f".format(aspectCorrX)}, ${"%.4f".format(aspectCorrY)}) stretch=$USE_STRETCH_TO_SQUARE")

        val monodepthInfo = if (!metricAnchors.isNullOrEmpty()) getLastMonodepthInfo() else null
        val scaleEstimation = if (!metricAnchors.isNullOrEmpty()) {
            val matched = MetricScaleEstimator.estimateFromMatchedMonodepth(metricAnchors, monodepthInfo, ::sampleMonodepthChannel)
            if (matched.isValid || matched.fallbackReason !in setOf("missing_monodepth_buffer", "insufficient_monodepth_pairings")) {
                matched
            } else {
                val gaussianDepthProxy = estimateGaussianDepthProxyUnits(params)
                MetricScaleEstimator.estimateFromGaussianDepthProxy(metricAnchors, gaussianDepthProxy)
            }
        } else {
            MetricScaleEstimator.EstimationResult(
                scale = 1f,
                isValid = false,
                fallbackReason = "no_metric_anchors",
                survivingAnchors = 0,
                coefficientOfVariation = Float.POSITIVE_INFINITY,
                arcoreMedianDepthMeters = Float.NaN,
                sharpMedianDepthUnits = Float.NaN,
                rawMedianRatio = Float.NaN,
                monodepthWidth = monodepthInfo?.width ?: 0,
                monodepthHeight = monodepthInfo?.height ?: 0,
                monodepthChannels = monodepthInfo?.channels ?: 0,
            )
        }
        val metricScale = if (scaleEstimation.isValid) scaleEstimation.scale else 1f
        LogUtil.i(
            "SHARP_METRIC_SCALE",
            "[estimate] anchors=${metricAnchors?.size ?: 0} surviving=${scaleEstimation.survivingAnchors} " +
                "sharpMedian=${scaleEstimation.sharpMedianDepthUnits} arcoreMedian=${scaleEstimation.arcoreMedianDepthMeters} " +
                "rawMedianRatio=${scaleEstimation.rawMedianRatio} scale=${scaleEstimation.scale} " +
                "valid=${scaleEstimation.isValid} reason=${scaleEstimation.fallbackReason ?: "ok"} " +
                "cv=${scaleEstimation.coefficientOfVariation} monodepth=${scaleEstimation.monodepthWidth}x${scaleEstimation.monodepthHeight}x${scaleEstimation.monodepthChannels}",
        )
        val roomFolder = File(File(context.filesDir, "sharp_rooms"), "room_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}").apply { mkdirs() }
        val plyFile = File(roomFolder, "room.ply")

        var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
        var minZ = Float.MAX_VALUE; var maxZ = -Float.MAX_VALUE
        var maxAbsX = 0f
        var maxAbsY = 0f
        var maxAbsZ = 0f

        val nativeStats = tryWritePlyNative(
            outputPath = plyFile.absolutePath,
            params = params,
            aspectCorrX = aspectCorrX,
            aspectCorrY = aspectCorrY,
            metricScale = metricScale,
        )
        if (nativeStats != null) {
            minX = nativeStats[0]
            maxX = nativeStats[1]
            minY = nativeStats[2]
            maxY = nativeStats[3]
            minZ = nativeStats[4]
            maxZ = nativeStats[5]
            maxAbsX = nativeStats[6]
            maxAbsY = nativeStats[7]
            maxAbsZ = nativeStats[8]
        } else {
            val progressReportEvery = (count / 8).coerceAtLeast(1)
            FileOutputStream(plyFile).use { fos ->
                val channel = fos.channel
                val header = "ply\nformat binary_little_endian 1.0\nelement vertex $count\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\n" +
                    (0 until 3).joinToString("") { "property float f_dc_$it\n" } + (0 until 45).joinToString("") { "property float f_rest_$it\n" } +
                    "property float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nend_header\n"
                channel.write(ByteBuffer.allocateDirect(header.length).apply { put(header.toByteArray()); flip() })
                plyBatch.clear()
                var batchedVertices = 0

                for (i in 0 until count) {
                    if (i > 0 && i % progressReportEvery == 0) {
                        report(0.91f + 0.09f * (i.toFloat() / count), "Writing your room file…", progressCallback)
                    }
                    val off = i * PARAMS_PER_GAUSSIAN
                    val rawX = params[off]
                    val rawY = params[off + 1]
                    val rawZ = params[off + 2]
                    val x = rawX * aspectCorrX * metricScale
                    val y = -(rawY * aspectCorrY) * metricScale
                    val z = -rawZ * metricScale
                    minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y); minZ = min(minZ, z); maxZ = max(maxZ, z)
                    maxAbsX = max(maxAbsX, abs(x))
                    maxAbsY = max(maxAbsY, abs(y))
                    maxAbsZ = max(maxAbsZ, abs(z))

                    plyBatch.putFloat(x).putFloat(y).putFloat(z).putFloat(0f).putFloat(0f).putFloat(0f)
                    val b = linearToSrgb(params[off + 11])
                    val g = linearToSrgb(params[off + 12])
                    val r = linearToSrgb(params[off + 13])
                    plyBatch.putFloat((r - 0.5f) / SH_C0).putFloat((g - 0.5f) / SH_C0).putFloat((b - 0.5f) / SH_C0)
                    plyBatch.put(zeroSHBytes)
                    plyBatch.putFloat(LOGIT_LUT[(params[off + 3] * 1023).toInt().coerceIn(0, 1023)])
                    plyBatch.putFloat(lnLut(max(params[off + 4] * 1.3f * metricScale, 0.001f)))
                        .putFloat(lnLut(max(params[off + 5] * 1.3f * metricScale, 0.001f)))
                        .putFloat(lnLut(max(params[off + 6] * 1.3f * metricScale, 0.001f)))

                    val rw = params[off + 7]; val rx = params[off + 8]; val ry = params[off + 9]; val rz = params[off + 10]
                    val m = sqrt(rw * rw + rx * rx + ry * ry + rz * rz).let { if (it > 1e-8f) 1f / it else 1f }
                    plyBatch.putFloat(rw * m).putFloat(rx * m).putFloat(ry * m).putFloat(rz * m)
                    batchedVertices++
                    if (batchedVertices == PLY_BATCH_SIZE || i == count - 1) {
                        plyBatch.flip()
                        while (plyBatch.hasRemaining()) channel.write(plyBatch)
                        plyBatch.clear()
                        batchedVertices = 0
                    }
                }
            }
        }
        val roomW = maxX - minX
        val roomH = maxY - minY
        val roomD = maxZ - minZ
        val centerX = (minX + maxX) * 0.5f
        val centerY = (minY + maxY) * 0.5f
        val centerZ = (minZ + maxZ) * 0.5f
        val looksNormalized =
            roomW <= 2.5f && roomH <= 2.5f && roomD <= 2.5f &&
                maxAbsX <= 1.5f && maxAbsY <= 1.5f && maxAbsZ <= 1.5f
        LogUtil.d(TAG, "[PLY] saved bbox: roomWidth=$roomW roomHeight=$roomH roomDepth=$roomD (raw coords)")
        LogUtil.i(
            "SHARP_ROOM_MEAS",
            "[ply_bbox] gaussians=$count W=$roomW H=$roomH D=$roomD " +
                "center=($centerX,$centerY,$centerZ) file=${plyFile.name} " +
                "(AABB in model space; WebGL viewer is authoritative for display dims)",
        )
        LogUtil.i(
            "SHARP_ROOM_MEAS",
            "[ply_space] min=($minX,$minY,$minZ) max=($maxX,$maxY,$maxZ) " +
                "maxAbs=($maxAbsX,$maxAbsY,$maxAbsZ) looksNormalized=$looksNormalized",
        )
        LogUtil.i(
            "SHARP_ROOM_MEAS",
            "[ply_aspect] aspectCorr=(${"%.4f".format(aspectCorrX)}, ${"%.4f".format(aspectCorrY)}) " +
                "metricScale=${"%.4f".format(metricScale)} " +
                "final W=${"%.3f".format(roomW)} H=${"%.3f".format(roomH)} D=${"%.3f".format(roomD)} " +
                "imgDims=${imgW}x${imgH}",
        )
        LogUtil.i(
            "SHARP_METRIC_SCALE",
            "[apply] beforeAabb=${"%.3f".format(roomW / metricScale)}x${"%.3f".format(roomH / metricScale)}x${"%.3f".format(roomD / metricScale)} " +
                "afterAabb=${"%.3f".format(roomW)}x${"%.3f".format(roomH)}x${"%.3f".format(roomD)} " +
                "metricScale=${"%.4f".format(metricScale)}",
        )
        if (roomW > 50f || roomH > 50f || roomD > 50f || roomW < 0.1f || roomH < 0.1f || roomD < 0.1f) {
            LogUtil.w(TAG, "[PLY] bbox may indicate scale/precision issue (expected room ~2–15 m)")
        }
        return StreamingResult(plyFile, plyFile, count, roomW, roomH, roomD, centerX, centerY, centerZ)
    }
}
