package com.furnit.android

import android.app.Application
import android.content.ComponentCallbacks2
import android.content.Intent
import android.os.Process
import com.furnit.android.services.SharpService
import com.furnit.android.utils.DebugLogger
import com.furnit.android.utils.LogUtil
import com.google.firebase.FirebaseApp

/**
 * FurnitApplication - Application class for initializing Firebase and crash reporting.
 * Production: no logging; on crash, user can send report (email) or copy details.
 * Registers onTrimMemory to release ML native caches and reduce OOM / CoroutineScheduler kills.
 */
class FurnitApplication : Application() {

    companion object {
        private const val TAG = "FurnitApp"
    }

    override fun onCreate() {
        super.onCreate()

        DebugLogger.init(this)
        LogUtil.init(this)
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
