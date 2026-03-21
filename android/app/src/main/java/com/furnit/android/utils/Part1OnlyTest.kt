package com.furnit.android.utils

import com.furnit.android.BuildConfig
import android.content.Context
import android.os.Process
import android.os.SystemClock
import android.util.Log
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs

/**
 * Part1-only test and warmup using a **persistent in-process** [Module]: load once per .pte path,
 * run **two** warmup [forward]s after load (Vulkan / driver cache), then **Run** reuses the module
 * for a single stats [forward] (same **PID** reuses cached [Module]; forward time still depends on backend).
 *
 * Call [releaseCachedPart1Module] if you replace `sharp_split_part1.pte` on disk or need RAM.
 *
 * **Vulkan perf:** [runTripleForwardBenchmark] logs `P1_BENCH` lines (same PID, same module, same patch)
 * to separate cold pipeline cost from steady-state — `adb logcat -s Part1Test:I | grep P1_BENCH`.
 */
object Part1OnlyTest {

    private const val TAG = "Part1Test"
    /** Grep: `adb logcat -d | grep P1_BENCH` — three timed forwards, same session. */
    const val P1_BENCH_MARKER = "P1_BENCH"
    /** Startup background warmup — adb: adb logcat -s ExecuTorchWarmup:I */
    const val WARMUP_TAG = "ExecuTorchWarmup"
    /**
     * Settings manual warmup + machine-readable status lines — adb:
     * `adb logcat -s Part1Warmup:I` or `adb logcat -d | grep WARMUP_STATUS`
     */
    const val PART1_WARMUP_LOG_TAG = "Part1Warmup"

    private const val PREFS = "furnit_prefs"
    private const val KEY_WARMUP_STATE = "part1_warmup_state"
    private const val KEY_WARMUP_DURATION_MS = "part1_warmup_duration_ms"
    private const val KEY_WARMUP_DETAIL = "part1_warmup_detail"
    private const val KEY_WARMUP_TIME_EPOCH_MS = "part1_warmup_time_epoch_ms"
    private const val KEY_STARTUP_WARMUP_ENABLED = "executorch_part1_startup_warmup_enabled"

    private val manualWarmupInProgress = AtomicBoolean(false)
    private val tripleBenchmarkInProgress = AtomicBoolean(false)
    private val startupWarmupSuppressed = AtomicBoolean(false)

    /** Same-process cache: survives Warmup → Run; lost on new PID or [releaseCachedPart1Module]. */
    private val sessionLock = Any()
    private var cachedModule: Module? = null
    private var cachedPtePath: String? = null

    private const val PATCH_SIZE = 384
    private const val PATCH_FLOATS = 1 * 3 * PATCH_SIZE * PATCH_SIZE // 442368

    /** Python export golden (first 8 floats); see `export_sharp_executorch_split4.py` part1-only output. */
    private val goldenTokensFirst8 = floatArrayOf(
        0.11831f, -0.08787f, -0.08292f, 0.07396f, 0.00907f, 0.02490f, 0.05221f, 0.03471f
    )
    private val goldenBlock5First8 = floatArrayOf(
        0.01150f, -0.01246f, -0.08190f, 0.01107f, -0.03018f, 0.09403f, 0.06268f, -0.00204f
    )
    private const val GOLDEN_FIRST8_TOLERANCE = 2e-4f

    data class WarmupResult(val success: Boolean, val durationMs: Long, val userMessage: String)

    private fun writeWarmupPrefs(context: Context, state: String, durationMs: Long, detail: String) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().apply {
            putString(KEY_WARMUP_STATE, state)
            putLong(KEY_WARMUP_DURATION_MS, durationMs)
            putString(KEY_WARMUP_DETAIL, detail)
            putLong(KEY_WARMUP_TIME_EPOCH_MS, System.currentTimeMillis())
            apply()
        }
    }

    @JvmStatic
    fun getWarmupStatusSummary(context: Context): String {
        val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val state = p.getString(KEY_WARMUP_STATE, null)
        if (state.isNullOrEmpty()) {
            return "Warmup status: not run yet — tap Warmup (same app session), then Run."
        }
        val ms = p.getLong(KEY_WARMUP_DURATION_MS, 0L)
        val detail = p.getString(KEY_WARMUP_DETAIL, "") ?: ""
        return when (state) {
            "running" -> "Warmup status: running… (Vulkan can take 10+ min; watch Part1Warmup logcat)"
            "completed" -> "Last warmup finished in ${ms / 1000}s (historical). Module still cached only until you Release or kill the app — tap Warmup again after Release."
            "failed" -> "Warmup status: failed — $detail"
            "skipped" -> "Warmup status: skipped — $detail"
            else -> "Warmup status: $state — $detail"
        }
    }

    /**
     * Drop cached [Module]. Next Warmup/Run will load + double-warm again.
     */
    @JvmStatic
    fun releaseCachedPart1Module() {
        synchronized(sessionLock) {
            try {
                cachedModule?.destroy()
            } catch (_: Throwable) { }
            cachedModule = null
            cachedPtePath = null
            Log.i(TAG, "session: released cached Part1 Module")
        }
    }

    @JvmStatic
    fun suppressStartupWarmupForCurrentProcess(reason: String) {
        if (startupWarmupSuppressed.compareAndSet(false, true)) {
            Log.i(WARMUP_TAG, "startup warmup suppressed for current process reason=$reason")
        }
    }

    private fun isStartupWarmupEnabled(context: Context): Boolean {
        return context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getBoolean(KEY_STARTUP_WARMUP_ENABLED, false)
    }

    private fun modelsDirs(context: Context): List<File> {
        val list = mutableListOf<File>()
        // etVulkan: models_vulkan first (push Vulkan .pte there via push_sharp_vulkan_only.sh)
        if (BuildConfig.EXECUTORCH_USE_VULKAN_AAR) {
            list.add(File(context.filesDir, "models_vulkan").also { it.mkdirs() })
            context.getExternalFilesDir("models_vulkan")?.let { list.add(it) }
        } else {
            // etCpu: models_cpu first (push via push_sharp_executorch_cpu_models.sh)
            list.add(File(context.filesDir, "models_cpu").also { it.mkdirs() })
            context.getExternalFilesDir("models_cpu")?.let { list.add(it) }
        }
        list.add(File(context.filesDir, "models").also { it.mkdirs() })
        context.getExternalFilesDir("models")?.let { list.add(it) }
        list.add(File("/data/local/tmp/furnit"))
        return list
    }

    /** Preferred Part1 filenames in order (first existing file is attempted first). */
    private fun part1PteCandidateNames(): List<String> {
        return if (BuildConfig.EXECUTORCH_USE_VULKAN_AAR) {
            listOf("sharp_split_part1_vulkan_fp32.pte", "sharp_split_part1_vulkan_fp16.pte", "sharp_split_part1.pte")
        } else {
            listOf("sharp_split_part1.pte", "sharp_split_part1_fp16.pte", "sharp_split_part1_int8.pte")
        }
    }

    /**
     * All existing Part1 .pte candidates (deduped by absolute path) in priority order.
     * Used to try multiple exports when runtime/export versions drift (common cause of 0x20 Resource not found).
     */
    private fun findPart1PteCandidates(context: Context): List<File> {
        val out = mutableListOf<File>()
        val seen = HashSet<String>()
        for (name in part1PteCandidateNames()) {
            val f = findFile(context, name) ?: continue
            val key = f.absolutePath
            if (seen.add(key)) out.add(f)
        }
        return out
    }

    private fun findFile(context: Context, filename: String): File? {
        for (dir in modelsDirs(context)) {
            val f = File(dir, filename)
            if (f.exists() && f.length() > 0) return f
        }
        return null
    }

    /**
     * Classify ExecuTorch/Vulkan failure for diagnostics. Log kind= in warmup/run paths.
     * No fallbacks — purely for clearer error messages and logcat filtering.
     */
    private fun classifyExecutorchFailure(e: Throwable): String {
        val msg = buildString {
            append(e.message ?: "")
            append(" ")
            append(e.cause?.message ?: "")
        }.lowercase()
        return when {
            "resource not found" in msg || "0x20" in msg -> "resource_not_found"
            "could not find shaderinfo" in msg ||
                "shaderregistry" in msg ||
                "get_shader_info" in msg -> "missing_vulkan_shader"
            "backend not found" in msg ||
                "failed to find backend" in msg ||
                "not registered" in msg -> "backend_missing"
            "vkcreate" in msg || "vk_error" in msg || "vulkanbackend" in msg ||
                "backendfailed" in msg || "backend failed" in msg ||
                "delegate init failed" in msg -> "vulkan_runtime_failure"
            else -> "unknown"
        }
    }

    private fun expectedPart1PteHint(): String {
        val names = part1PteCandidateNames().joinToString(" | ")
        val dirHint = if (BuildConfig.EXECUTORCH_USE_VULKAN_AAR) "models_vulkan (or models)" else "models_cpu (or models)"
        return "$names in $dirHint"
    }

    /** First matching path wins ([modelsDirs] order); log so we never debug the wrong artifact. */
    private fun logPart1Artifacts(label: String, pte: File, patch: File) {
        Log.i(
            TAG,
            "PART1_ARTIFACT pte path=${pte.absolutePath} size=${pte.length()} mtime=${pte.lastModified()} label=$label"
        )
        Log.i(
            TAG,
            "PART1_ARTIFACT patch path=${patch.absolutePath} size=${patch.length()} mtime=${patch.lastModified()} label=$label"
        )
    }

    private fun maxAbsDiffFirstN(actual: FloatArray, expected: FloatArray, count: Int): Float {
        var maxDiff = 0f
        for (i in 0 until count) {
            maxDiff = maxOf(maxDiff, abs(actual[i] - expected[i]))
        }
        return maxDiff
    }

    private fun readF32Bin(file: File, expectedFloats: Int): FloatArray {
        val bytes = expectedFloats * 4
        if (file.length() < bytes) throw IllegalArgumentException("${file.name} too small: ${file.length()} < $bytes")
        val buf = ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN)
        RandomAccessFile(file, "r").use { raf ->
            raf.channel.read(buf)
        }
        buf.flip()
        val arr = FloatArray(expectedFloats)
        buf.asFloatBuffer().get(arr)
        return arr
    }

    /**
     * Try each candidate .pte until one successfully loads + runs 2x warmup forward.
     * Keeps the winning [Module] cached for subsequent Run / benchmark.
     */
    private fun ensureSessionLoadedAndDoubleWarmupWithFallback(
        context: Context,
        patchBin: File,
        logLabel: String
    ): File {
        val candidates = findPart1PteCandidates(context)
        if (candidates.isEmpty()) {
            throw IllegalStateException("No Part1 .pte found (expected ${expectedPart1PteHint()})")
        }

        var lastError: Throwable? = null
        for (pte in candidates) {
            try {
                logPart1Artifacts(logLabel, pte, patchBin)
                ensureSessionLoadedAndDoubleWarmup(pte, patchBin, Module.LOAD_MODE_MMAP)
                return pte
            } catch (e: Throwable) {
                lastError = e
                if (e is UnsatisfiedLinkError) throw e
                val kind = classifyExecutorchFailure(e)
                if (kind == "resource_not_found") {
                    try {
                        Log.w(TAG, "Part1 retry with LOAD_MODE_FILE after resource_not_found pte=${pte.name}")
                        ensureSessionLoadedAndDoubleWarmup(pte, patchBin, Module.LOAD_MODE_FILE)
                        return pte
                    } catch (e2: Throwable) {
                        lastError = e2
                        val kind2 = classifyExecutorchFailure(e2)
                        Log.e(
                            TAG,
                            "Part1 retry failed kind=$kind2 pte=${pte.name} path=${pte.absolutePath} err=${e2.message ?: e2}",
                            e2
                        )
                    }
                }
                Log.e(
                    TAG,
                    "Part1 warmup candidate failed kind=$kind pte=${pte.name} path=${pte.absolutePath} err=${e.message ?: e}",
                    e
                )
            }
        }
        throw (lastError ?: IllegalStateException("Part1 warmup failed (no error captured)"))
    }

    private fun tensorStats(arr: FloatArray, name: String): String {
        if (arr.isEmpty()) return "$name: empty"
        var min = arr[0]
        var max = arr[0]
        var sum = 0.0
        for (x in arr) {
            if (x < min) min = x
            if (x > max) max = x
            sum += x
        }
        val mean = sum / arr.size
        val first8 = arr.take(8)
        return "$name shape=${arr.size} min=$min max=$max mean=$mean first8=$first8"
    }

    private fun buildPatchTensor(patchData: FloatArray): Tensor {
        return Tensor.fromBlob(patchData, longArrayOf(1, 3, PATCH_SIZE.toLong(), PATCH_SIZE.toLong()))
    }

    /**
     * Load [Module] once per [pteFile] path, run **two** [forward] passes (driver / Vulkan cache).
     * Cache hit: no disk read. Cache miss: read patch once outside lock, then load + warm under lock.
     */
    private fun ensureSessionLoadedAndDoubleWarmup(pteFile: File, patchBin: File, loadMode: Int = Module.LOAD_MODE_MMAP) {
        val path = pteFile.absolutePath
        synchronized(sessionLock) {
            if (cachedModule != null && cachedPtePath == path) {
                Log.d(TAG, "session: cache_hit pte_path=$path skip_load skip_warmup same_PID")
                return
            }
        }
        val patchDataForWarmup = readF32Bin(patchBin, PATCH_FLOATS)
        synchronized(sessionLock) {
            if (cachedModule != null && cachedPtePath == path) {
                Log.d(TAG, "session: cache_hit_after_read pte_path=$path skip_load (other_thread_filled_cache)")
                return
            }
            Log.d(TAG, "session: cache_miss pte_path=$path will_Module_load_and_2x_warmup")
            try {
                cachedModule?.destroy()
            } catch (_: Throwable) { }
            cachedModule = null
            cachedPtePath = null

            ExecutorchNativeLoader.loadForJavaModule()
            Log.d(TAG, "session: Module.load($path, load_mode=$loadMode)…")
            val loadedModule = Module.load(path, loadMode)
            try {
                repeat(2) { warmupIndex ->
                    Log.d(TAG, "session: warmup forward ${warmupIndex + 1}/2…")
                    val out = loadedModule.forward(EValue.from(buildPatchTensor(patchDataForWarmup)))
                    if (out == null || out.size < 2) {
                        throw IllegalStateException("warmup forward returned ${out?.size ?: 0} outputs")
                    }
                }
                cachedModule = loadedModule
                cachedPtePath = path
                Log.d(TAG, "session: double warmup done; Module kept alive for Run")
            } catch (e: Throwable) {
                try {
                    loadedModule.destroy()
                } catch (_: Throwable) { }
                throw e
            }
        }
    }

    /**
     * One [forward] using cached [Module]; call [ensureSessionLoadedAndDoubleWarmup] first.
     * [patchData] length must be [PATCH_FLOATS] (caller reads disk once if needed).
     */
    private fun forwardWithCachedModule(patchData: FloatArray): Array<EValue> {
        val input = buildPatchTensor(patchData)
        synchronized(sessionLock) {
            val module = cachedModule ?: throw IllegalStateException("Part1 Module not loaded — run Warmup first")
            val outputs = module.forward(EValue.from(input))
            if (outputs == null || outputs.size < 2) {
                throw IllegalStateException("forward returned ${outputs?.size ?: 0} outputs (expected 2)")
            }
            return outputs
        }
    }

    /**
     * Single [forward] duration (ms) only — timing excludes disk read; uses one [patchData] blob (same patch 3×).
     */
    private fun forwardOnceDurationMs(patchData: FloatArray): Long {
        val input = buildPatchTensor(patchData)
        synchronized(sessionLock) {
            val module = cachedModule ?: throw IllegalStateException("Part1 Module not loaded")
            val t0 = SystemClock.elapsedRealtime()
            val outputs = module.forward(EValue.from(input))
            val elapsedMs = SystemClock.elapsedRealtime() - t0
            if (outputs == null || outputs.size < 2) {
                throw IllegalStateException("forward returned ${outputs?.size ?: 0} outputs (expected 2)")
            }
            return elapsedMs
        }
    }

    /**
     * Part1 only: same process, same cached [Module], same `part1_test_patch_f32.bin` bytes, **three** timed
     * [forward]s in a row. Does **not** replace Warmup — calls [ensureSessionLoadedAndDoubleWarmup] first if cold
     * (load + 2× warmup not included in the three timings).
     *
     * Interpretation: if `duration_ms_2` and `duration_ms_3` stay huge vs run1, steady-state Vulkan cost dominates
     * (layout / attention / repacking), not one-shot pipeline creation.
     *
     * Call from a **background** dispatcher. Logs at **INFO** for easy capture: `grep P1_BENCH`.
     */
    @JvmStatic
    fun runTripleForwardBenchmark(context: Context): String {
        if (!tripleBenchmarkInProgress.compareAndSet(false, true)) {
            return "Part1 benchmark: already running"
        }
        try {
            val app = context.applicationContext
            val candidates = findPart1PteCandidates(app)
            val patchBin = findFile(app, "part1_test_patch_f32.bin")
            if (candidates.isEmpty()) {
                val expected = expectedPart1PteHint()
                Log.e(TAG, "$P1_BENCH_MARKER missing $expected")
                return "Part1 benchmark: $expected not found"
            }
            if (patchBin == null) {
                Log.e(TAG, "$P1_BENCH_MARKER missing part1_test_patch_f32.bin")
                return "Part1 benchmark: part1_test_patch_f32.bin not found"
            }
            ExecutorchNativeLoader.loadForJavaModule()
            val pid = Process.myPid()
            Log.i(TAG, "$P1_BENCH_MARKER pid=$pid pte_candidates=${candidates.joinToString { it.name }}")
            Log.i(TAG, "$P1_BENCH_MARKER shape=[1,3,$PATCH_SIZE,$PATCH_SIZE] dtype=float32 same_patch_blob=1 timed_forwards=3")
            Log.i(
                TAG,
                "$P1_BENCH_MARKER note=session_ensure_may_load_and_2x_warmup_if_cold_timings_below_are_only_the_3_runs"
            )

            val tEnsure = SystemClock.elapsedRealtime()
            val chosenPte = ensureSessionLoadedAndDoubleWarmupWithFallback(app, patchBin, "benchmark")
            val ensureMs = SystemClock.elapsedRealtime() - tEnsure
            Log.i(TAG, "$P1_BENCH_MARKER using_pte=${chosenPte.name}")
            Log.i(TAG, "$P1_BENCH_MARKER session_ensure_ms=$ensureMs (includes_load_2x_warmup_if_cold)")

            val patchData = readF32Bin(patchBin, PATCH_FLOATS)
            val durationsMs = LongArray(3)
            for (i in 0 until 3) {
                Log.i(TAG, "$P1_BENCH_MARKER timed_forward ${i + 1}/3 start")
                durationsMs[i] = forwardOnceDurationMs(patchData)
                Log.i(TAG, "$P1_BENCH_MARKER timed_forward ${i + 1}/3 duration_ms=${durationsMs[i]}")
            }

            val d1 = durationsMs[0]
            val d2 = durationsMs[1]
            val d3 = durationsMs[2]
            val ratio21 = if (d1 > 0) d2.toDouble() / d1 else 0.0
            val ratio31 = if (d1 > 0) d3.toDouble() / d1 else 0.0
            Log.i(
                TAG,
                "$P1_BENCH_MARKER summary duration_ms=[$d1,$d2,$d3] ratio_2_over_1=$ratio21 ratio_3_over_1=$ratio31"
            )
            Log.i(
                TAG,
                "$P1_BENCH_MARKER interpret all_three_high=steady_exec_cost run2_run3_much_lower_than_run1=pipeline_cache"
            )

            return "Part1 benchmark OK: ${d1}ms, ${d2}ms, ${d3}ms (see logcat P1_BENCH)"
        } catch (e: Throwable) {
            Log.e(TAG, "$P1_BENCH_MARKER failed", e)
            return "Part1 benchmark failed: ${e.message ?: e}"
        } finally {
            tripleBenchmarkInProgress.set(false)
        }
    }

    @JvmStatic
    fun runWarmupFromSettings(context: Context): WarmupResult {
        if (!manualWarmupInProgress.compareAndSet(false, true)) {
            Log.w(PART1_WARMUP_LOG_TAG, "WARMUP_STATUS state=rejected reason=already_running")
            return WarmupResult(false, 0L, "Warmup already running")
        }
        try {
            val app = context.applicationContext
            writeWarmupPrefs(app, "running", 0L, "In progress…")
            Log.i(PART1_WARMUP_LOG_TAG, "WARMUP_STATUS state=running source=settings")

            val candidates = findPart1PteCandidates(app)
            val patchBin = findFile(app, "part1_test_patch_f32.bin")
            if (candidates.isEmpty() || patchBin == null) {
                val msg = "missing ${expectedPart1PteHint()} and/or part1_test_patch_f32.bin"
                writeWarmupPrefs(app, "skipped", 0L, msg)
                Log.w(PART1_WARMUP_LOG_TAG, "WARMUP_STATUS state=skipped reason=$msg")
                return WarmupResult(false, 0L, "Warmup skipped: $msg")
            }

            val t0 = SystemClock.elapsedRealtime()
            return try {
                Log.i(WARMUP_TAG, "settings warmup: load + 2x forward (Module stays in memory)…")
                val chosenPte = ensureSessionLoadedAndDoubleWarmupWithFallback(app, patchBin, "settings_warmup")
                val ms = SystemClock.elapsedRealtime() - t0
                writeWarmupPrefs(app, "completed", ms, "OK")
                Log.i(PART1_WARMUP_LOG_TAG, "WARMUP_STATUS state=completed duration_ms=$ms source=settings session=cached pte=${chosenPte.name}")
                Log.i(WARMUP_TAG, "settings Part1 warmup OK in ${ms}ms (Module cached for Run)")
                WarmupResult(true, ms, "Warmup completed in ${ms / 1000}s — Module cached until Release or app kill (same PID).")
            } catch (e: Throwable) {
                val ms = SystemClock.elapsedRealtime() - t0
                val kind = classifyExecutorchFailure(e)
                val err = e.message ?: e.toString()
                writeWarmupPrefs(app, "failed", ms, err)
                Log.e(PART1_WARMUP_LOG_TAG, "WARMUP_STATUS state=failed kind=$kind duration_ms=$ms error=$err", e)
                WarmupResult(false, ms, "Warmup failed: $err")
            }
        } finally {
            manualWarmupInProgress.set(false)
        }
    }

    @JvmStatic
    fun scheduleStartupWarmup(context: Context) {
        val app = context.applicationContext
        if (!isStartupWarmupEnabled(app)) {
            Log.d(WARMUP_TAG, "skip startup warmup (disabled; enable via pref $KEY_STARTUP_WARMUP_ENABLED)")
            Log.d(PART1_WARMUP_LOG_TAG, "WARMUP_STATUS state=skipped source=startup reason=disabled")
            return
        }
        if (startupWarmupSuppressed.get()) {
            Log.d(WARMUP_TAG, "skip startup warmup (suppressed by full inference)")
            Log.d(PART1_WARMUP_LOG_TAG, "WARMUP_STATUS state=skipped source=startup reason=suppressed")
            return
        }
        Thread(
            {
                if (startupWarmupSuppressed.get()) {
                    Log.d(WARMUP_TAG, "skip startup warmup thread (suppressed by full inference)")
                    Log.d(PART1_WARMUP_LOG_TAG, "WARMUP_STATUS state=skipped source=startup reason=suppressed")
                    return@Thread
                }
                val candidates = findPart1PteCandidates(app)
                val patchBin = findFile(app, "part1_test_patch_f32.bin")
                if (candidates.isEmpty() || patchBin == null) {
                    Log.d(WARMUP_TAG, "skip startup warmup (no models on device)")
                    Log.d(PART1_WARMUP_LOG_TAG, "WARMUP_STATUS state=skipped source=startup reason=no_models_on_device")
                    return@Thread
                }
                if (startupWarmupSuppressed.get()) {
                    Log.d(WARMUP_TAG, "abort startup warmup before load (suppressed by full inference)")
                    Log.d(PART1_WARMUP_LOG_TAG, "WARMUP_STATUS state=skipped source=startup reason=suppressed")
                    return@Thread
                }
                val t0 = SystemClock.elapsedRealtime()
                try {
                    Log.i(WARMUP_TAG, "startup: Part1 load + 2x forward (session cache)…")
                    val chosenPte = ensureSessionLoadedAndDoubleWarmupWithFallback(app, patchBin, "startup_warmup")
                    val ms = SystemClock.elapsedRealtime() - t0
                    Log.i(WARMUP_TAG, "startup Part1 session ready in ${ms}ms")
                    Log.i(PART1_WARMUP_LOG_TAG, "WARMUP_STATUS state=completed duration_ms=$ms source=startup session=cached pte=${chosenPte.name}")
                } catch (e: Throwable) {
                    val ms = SystemClock.elapsedRealtime() - t0
                    val kind = classifyExecutorchFailure(e)
                    val err = e.message ?: e.toString()
                    Log.e(WARMUP_TAG, "startup warmup failed after ${ms}ms", e)
                    Log.e(PART1_WARMUP_LOG_TAG, "WARMUP_STATUS state=failed kind=$kind duration_ms=$ms source=startup error=$err")
                }
            },
            "ExecuTorchPart1Warmup"
        ).apply {
            isDaemon = true
            priority = Thread.MIN_PRIORITY
            start()
        }
    }

    /**
     * Part1-only test. Call from a **background** thread. Reuses cached [Module] after Warmup.
     * Logs **PART1_RUN** (`session_ensure_ms`, `forward_only_ms`): forward timing excludes patch disk read,
     * session ensure, and output stats. See **PART1_GOLDEN** for first-8 self-check vs Python export.
     */
    @JvmStatic
    fun run(context: Context): String {
        val candidates = findPart1PteCandidates(context)
        val patchBin = findFile(context, "part1_test_patch_f32.bin")
        if (candidates.isEmpty()) {
            val expected = expectedPart1PteHint()
            Log.e(TAG, "$expected not found")
            return "Part1 test: $expected not found"
        }
        if (patchBin == null) {
            Log.e(TAG, "part1_test_patch_f32.bin not found")
            return "Part1 test: part1_test_patch_f32.bin not found"
        }
        return try {
            val patchData = readF32Bin(patchBin, PATCH_FLOATS)
            val pid = Process.myPid()
            Log.d(TAG, "input patch shape=[1,3,$PATCH_SIZE,$PATCH_SIZE] dtype=float32")
            Log.d(TAG, "input first 8: ${patchData.take(8)}")

            Log.d(TAG, "ExecuTorch native + session (load+2x warmup if needed, else reuse)…")
            Log.w(
                TAG,
                "Vulkan Part1: first load / each forward can take many minutes on some devices (not just registration). " +
                    "Use PART1_RUN forward_only_ms + P1_BENCH; portable .pte for CPU baseline."
            )
            val tSession = SystemClock.elapsedRealtime()
            val chosenPte = ensureSessionLoadedAndDoubleWarmupWithFallback(context, patchBin, "run")
            val sessionEnsureMs = SystemClock.elapsedRealtime() - tSession
            Log.d(TAG, "session ensure finished in ${sessionEnsureMs}ms (0 if already cached)")
            Log.i(TAG, "PART1_RUN session_ensure_ms=$sessionEnsureMs pid=$pid pte=${chosenPte.name}")

            val tFwd = SystemClock.elapsedRealtime()
            val outputs = forwardWithCachedModule(patchData)
            val fwdMs = SystemClock.elapsedRealtime() - tFwd
            Log.d(TAG, "forward_only_ms=$fwdMs (excludes session_ensure, patch disk read, output stats logging)")
            Log.i(TAG, "PART1_RUN forward_only_ms=$fwdMs pid=$pid")

            val tokensTensor = outputs[0].toTensor()
            val block5Tensor = outputs[1].toTensor()
            val tokensArr = tokensTensor.getDataAsFloatArray()
            val block5Arr = block5Tensor.getDataAsFloatArray()

            Log.d(TAG, "output0 (tokens): ${tensorStats(tokensArr, "tokens")}")
            Log.d(TAG, "output1 (block5): ${tensorStats(block5Arr, "block5")}")

            val diffTok = maxAbsDiffFirstN(tokensArr, goldenTokensFirst8, 8)
            val diffB5 = maxAbsDiffFirstN(block5Arr, goldenBlock5First8, 8)
            val goldenPass = diffTok <= GOLDEN_FIRST8_TOLERANCE && diffB5 <= GOLDEN_FIRST8_TOLERANCE
            Log.i(
                TAG,
                "PART1_GOLDEN first8_tokens_max_abs_diff=$diffTok first8_block5_max_abs_diff=$diffB5 " +
                    "tol=$GOLDEN_FIRST8_TOLERANCE pass=$goldenPass"
            )
            Log.d(
                TAG,
                "Python golden ref: tokens first8=${goldenTokensFirst8.contentToString()} block5 first8=${goldenBlock5First8.contentToString()}"
            )

            Log.d(
                TAG,
                "PART1_RUN forward completed; Vulkan delegate usage is not proven here — confirm via .pte inspection + backend logs."
            )
            "Part1 test OK (forward_only_ms=$fwdMs). golden_first8_pass=$goldenPass"
        } catch (e: Throwable) {
            Log.e(TAG, "Part1 test failed", e)
            if (e is UnsatisfiedLinkError) {
                return "Part1 test: native load failed (${e.message}). Reinstall APK / check ABI."
            }
            val kind = classifyExecutorchFailure(e)
            val msg = e.message ?: ""
            val causeMsg = e.cause?.message ?: ""
            val combined = "$msg $causeMsg".lowercase()
            Log.e(TAG, "PART1_ERROR kind=$kind msg=$combined")
            return when (kind) {
                "resource_not_found" ->
                    "Part1 test: ExecuTorch/Vulkan resource not found during forward. Usually export/runtime mismatch or missing Vulkan shader in AAR. Rebuild ExecuTorch from source with EXECUTORCH_BUILD_VULKAN=ON, or re-export .pte from same ExecuTorch commit as runtime."
                "missing_vulkan_shader" ->
                    "Part1 test: Vulkan shader missing (ShaderRegistry). Build ExecuTorch from source with EXECUTORCH_BUILD_VULKAN=ON."
                "backend_missing" ->
                    "Part1 test: Vulkan backend not registered / not found. Check AAR and load order."
                "vulkan_runtime_failure" ->
                    "Part1 test: Vulkan runtime failed during forward. See logcat."
                else -> "Part1 test failed: $msg"
            }
        }
    }
}
