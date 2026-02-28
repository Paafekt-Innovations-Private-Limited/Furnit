package com.furnit.android

import android.app.Application
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.os.Process
import android.util.Log
import com.furnit.android.utils.DebugLogger
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
        installCrashHandler()

        try {
            FirebaseApp.initializeApp(this)
            if (isDebugBuild()) {
                Log.d(TAG, "Firebase initialized successfully")
            }
        } catch (e: Exception) {
            if (isDebugBuild()) {
                Log.e(TAG, "Failed to initialize Firebase", e)
            }
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

    private fun isDebugBuild(): Boolean {
        return (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0
    }
}
