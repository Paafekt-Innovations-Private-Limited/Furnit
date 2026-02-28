package com.furnit.android.utils

import android.content.Context
import android.content.pm.ApplicationInfo
import android.util.Log

/**
 * Release-safe logging: no-op in production (release builds).
 * Use this instead of Log.* so nothing is written to logcat when the app is live.
 */
object LogUtil {
    private var appContext: Context? = null

    fun init(context: Context) {
        appContext = context.applicationContext
    }

    private val isDebugBuild: Boolean
        get() {
            val ctx = appContext ?: return false
            return (ctx.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0
        }

    fun d(tag: String, message: String) {
        if (isDebugBuild) Log.d(tag, message)
    }

    fun i(tag: String, message: String) {
        if (isDebugBuild) Log.i(tag, message)
    }

    fun w(tag: String, message: String, throwable: Throwable? = null) {
        if (isDebugBuild) {
            if (throwable != null) Log.w(tag, message, throwable)
            else Log.w(tag, message)
        }
    }

    fun e(tag: String, message: String, throwable: Throwable? = null) {
        if (isDebugBuild) {
            if (throwable != null) Log.e(tag, message, throwable)
            else Log.e(tag, message)
        }
    }
}
