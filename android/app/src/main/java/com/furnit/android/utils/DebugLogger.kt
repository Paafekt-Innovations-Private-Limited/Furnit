package com.furnit.android.utils

import android.content.Context
import android.content.SharedPreferences
import android.util.Log

/**
 * Debug logger utility that respects the debug mode setting.
 * Matches iOS Logger.swift implementation.
 *
 * Usage:
 *   DebugLogger.init(context)
 *   DebugLogger.d("Tag", "Message")
 *   DebugLogger.log("Message with emoji prefix")
 */
object DebugLogger {
    private const val TAG = "Furnit"
    private const val DEBUG_MODE_KEY = "debug_mode"
    private const val PREFS_NAME = "furnit_prefs"

    private var prefs: SharedPreferences? = null
    private var appContext: Context? = null

    /**
     * Initialize the logger with a context.
     * Should be called early in Application or Activity lifecycle.
     */
    fun init(context: Context) {
        appContext = context.applicationContext
        prefs = appContext!!.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    /**
     * Check if debug mode is enabled
     */
    val isDebugMode: Boolean
        get() = prefs?.getBoolean(DEBUG_MODE_KEY, false) ?: false

    /**
     * When SHARP ExecuTorch native (C++) should emit verbose logcat (debuggable build + debug_mode).
     * Kotlin syncs this to JNI before pipeline / preload calls.
     */
    val isSharpNativeVerboseEnabled: Boolean
        get() = isLoggingEnabled && isDebugMode

    /**
     * Set debug mode
     */
    fun setDebugMode(enabled: Boolean) {
        prefs?.edit()?.putBoolean(DEBUG_MODE_KEY, enabled)?.apply()
        log("Debug mode ${if (enabled) "enabled" else "disabled"}")
    }

    /** True when we are allowed to log (debug build only; no logging in production). */
    private val isLoggingEnabled: Boolean
        get() {
            val ctx = appContext ?: return false
            return (ctx.applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0
        }

    /**
     * Log a debug message (only in debug builds and when debug mode is enabled)
     */
    fun log(message: String) {
        if (isLoggingEnabled && isDebugMode) {
            Log.d(TAG, message)
        }
    }

    /**
     * Log a debug message with a custom tag (only in debug builds and when debug mode is enabled)
     */
    fun d(tag: String, message: String) {
        if (isLoggingEnabled && isDebugMode) {
            Log.d(tag, message)
        }
    }

    /**
     * Log an info message with a custom tag (only in debug builds and when debug mode is enabled)
     */
    fun i(tag: String, message: String) {
        if (isLoggingEnabled && isDebugMode) {
            Log.i(tag, message)
        }
    }

    /**
     * Log a warning message with a custom tag (only in debug builds and when debug mode is enabled)
     */
    fun w(tag: String, message: String) {
        if (isLoggingEnabled && isDebugMode) {
            Log.w(tag, message)
        }
    }

    /**
     * Log an error message (only in debug builds; no logging in production)
     */
    fun e(tag: String, message: String, throwable: Throwable? = null) {
        if (isLoggingEnabled) {
            if (throwable != null) {
                Log.e(tag, message, throwable)
            } else {
                Log.e(tag, message)
            }
        }
    }

    /**
     * Log an error only when Settings debug_mode is on (and debuggable build).
     * Use for diagnostic errors that should not spam logcat when Debug mode is off.
     */
    fun eDebugMode(tag: String, message: String, throwable: Throwable? = null) {
        if (!isLoggingEnabled || !isDebugMode) return
        if (throwable != null) {
            Log.e(tag, message, throwable)
        } else {
            Log.e(tag, message)
        }
    }

    /**
     * Log with emoji prefix (matches iOS logging style)
     */
    fun camera(message: String) = log("📷 $message")
    fun model(message: String) = log("🏠 $message")
    fun save(message: String) = log("💾 $message")
    fun load(message: String) = log("📂 $message")
    fun reset(message: String) = log("🔄 $message")
    fun warning(message: String) = log("⚠️ $message")
    fun success(message: String) = log("✅ $message")
    fun error(message: String) = log("❌ $message")
    fun ai(message: String) = log("🤖 $message")
    fun measure(message: String) = log("📐 $message")
}

/**
 * Global debug log function for convenience
 * Similar to iOS logDebug() global function
 */
fun logDebug(message: String) {
    DebugLogger.log(message)
}
