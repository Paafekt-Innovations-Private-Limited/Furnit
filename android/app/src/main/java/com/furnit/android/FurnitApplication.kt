package com.furnit.android

import android.app.Application
import android.content.Intent
import android.os.Process
import com.furnit.android.utils.DebugLogger
import com.furnit.android.utils.LogUtil
import com.google.firebase.FirebaseApp

/**
 * FurnitApplication - Application class for initializing Firebase and crash reporting.
 * Production: no logging; on crash, user can send report (email) or copy details.
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

        try {
            FirebaseApp.initializeApp(this)
            LogUtil.d(TAG, "Firebase initialized successfully")
        } catch (e: Exception) {
            LogUtil.e(TAG, "Failed to initialize Firebase", e)
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
