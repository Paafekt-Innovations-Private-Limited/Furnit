package com.furnit.android

import android.app.Application
import android.content.ComponentCallbacks2
import android.content.Intent
import android.os.Process
import com.furnit.android.services.BackendConfig
import com.furnit.android.services.ExecutorchInt8Sharp
import com.furnit.android.services.SharpExecuTorchSplitModelNames
import com.furnit.android.services.SharpService
import com.furnit.android.utils.DebugLogger
import com.furnit.android.utils.DeviceHeuristics
import com.furnit.android.utils.ExecutorchNativeLoader
import com.furnit.android.utils.LogUtil
import com.furnit.android.utils.Part1OnlyTest
import com.google.firebase.FirebaseApp

/**
 * FurnitApplication - Application class for initializing Firebase and crash reporting.
 * Production: no logging; on crash, user can send report (email) or copy details.
 * Registers onTrimMemory to release ML native caches and reduce OOM / CoroutineScheduler kills.
 * At startup: loads ExecuTorch native libs; background Part1 warmup is opt-in only (logcat ExecuTorchWarmup).
 */
class FurnitApplication : Application() {

    companion object {
        private const val TAG = "FurnitApp"
        private const val PREFS = "furnit_prefs"
        private const val KEY_PART12_CPU = "executorch_int8_part12_on_cpu"
        /** One-shot so we only apply the Pixel default when the key has never been set. */
        private const val KEY_PART12_PIXEL_MIGRATION = "executorch_int8_part12_pixel_default_v1"
    }

    /**
     * Pixel / Tensor: Vulkan Part1+2 often hits GPU timeout or device lost. Default "Part1+2 on CPU" ON when the user
     * has never changed it, so hybrid mode is used as soon as INT8 sidecars exist (models_vulkan or models_cpu).
     */
    private fun migratePixelHybridPart12Default() {
        if (!DeviceHeuristics.isGooglePixelFamily()) return
        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        if (prefs.getBoolean(KEY_PART12_PIXEL_MIGRATION, false)) return
        val editor = prefs.edit().putBoolean(KEY_PART12_PIXEL_MIGRATION, true)
        if (!prefs.contains(KEY_PART12_CPU)) {
            editor.putBoolean(KEY_PART12_CPU, true)
            LogUtil.i(
                TAG,
                "Pixel-family device: defaulting Part1+2 on CPU (hybrid). Push ${SharpExecuTorchSplitModelNames.PART1_INT8} + " +
                    "${SharpExecuTorchSplitModelNames.PART2_INT8} to files/models_vulkan/ (with other Vulkan .pte) if inference still fails.",
            )
        }
        editor.apply()
    }

    /**
     * Friend APKs bundle .pte under assets/models_cpu + assets/models_vulkan. On first launch, internal dirs are
     * empty — copy from assets (and sync scoped external) on a background thread so SHARP works without adb push.
     */
    private fun scheduleHydrateSharpModelsFromApkAssetsIfNeeded() {
        if (!BackendConfig.ENABLE_EXECUTORCH_INT8) return
        Thread(
            {
                try {
                    val sharp = ExecutorchInt8Sharp.getInstance(this@FurnitApplication)
                    if (!sharp.internalSplitSharpModelsAbsent()) return@Thread
                    sharp.hydrateBundledAndExternalModels()
                    LogUtil.d(TAG, "Startup: hydrated SHARP models from APK assets / external (if any)")
                } catch (e: Throwable) {
                    LogUtil.d(TAG, "Startup SHARP model hydrate skipped: ${e.message}")
                }
            },
            "SharpAssetHydrate",
        ).start()
    }

    override fun onCreate() {
        super.onCreate()

        DebugLogger.init(this)
        LogUtil.init(this)

        migratePixelHybridPart12Default()
        scheduleHydrateSharpModelsFromApkAssetsIfNeeded()

        // ExecuTorch: register Vulkan/backend early. Background Part1 warmup is opt-in and disabled by default.
        try {
            ExecutorchNativeLoader.loadForJavaModule()
            LogUtil.d(TAG, "ExecuTorch native libs loaded at startup (core → executorch → executorch_jni)")
        } catch (e: UnsatisfiedLinkError) {
            LogUtil.w(TAG, "ExecuTorch startup native load failed (non-fatal): ${e.message}")
        }
        Part1OnlyTest.scheduleStartupWarmup(this)

        installCrashHandler()
        registerComponentCallbacks(componentCallbacks2)

        try {
            FirebaseApp.initializeApp(this)
            LogUtil.d(TAG, "Firebase initialized successfully")
        } catch (e: Exception) {
            LogUtil.e(TAG, "Failed to initialize Firebase", e)
        }
    }

    private val componentCallbacks2 = object : ComponentCallbacks2 {
        override fun onTrimMemory(level: Int) {
            when (level) {
                ComponentCallbacks2.TRIM_MEMORY_RUNNING_MODERATE,
                ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW,
                ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL,
                ComponentCallbacks2.TRIM_MEMORY_UI_HIDDEN,
                ComponentCallbacks2.TRIM_MEMORY_BACKGROUND,
                ComponentCallbacks2.TRIM_MEMORY_MODERATE,
                ComponentCallbacks2.TRIM_MEMORY_COMPLETE -> {
                    try {
                        SharpService.getInstance(this@FurnitApplication).releaseNativeCaches()
                        LogUtil.d(TAG, "onTrimMemory($level): released native caches")
                    } catch (_: Throwable) { }
                }
            }
        }
        override fun onConfigurationChanged(newConfig: android.content.res.Configuration) {}
        override fun onLowMemory() {
            try {
                SharpService.getInstance(this@FurnitApplication).releaseNativeCaches()
                LogUtil.d(TAG, "onLowMemory: released native caches")
            } catch (_: Throwable) { }
        }
    }

    private fun installCrashHandler() {
        Thread.setDefaultUncaughtExceptionHandler { _, throwable ->
            val message = throwable.message ?: throwable.toString()
            val stackTrace = throwable.stackTraceToString()
            val intent = Intent(this, CrashReportActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                putExtra(CrashReportActivity.EXTRA_CRASH_MESSAGE, message)
                putExtra(CrashReportActivity.EXTRA_CRASH_STACKTRACE, stackTrace)
            }
            try {
                startActivity(intent)
            } catch (_: Exception) { }
            Process.killProcess(Process.myPid())
            Runtime.getRuntime().exit(0)
        }
    }

}
