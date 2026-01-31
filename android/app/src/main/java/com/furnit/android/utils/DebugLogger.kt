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

    /**
     * Initialize the logger with a context.
     * Should be called early in Application or Activity lifecycle.
     */
    fun init(context: Context) {
        prefs = context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    /**
     * Check if debug mode is enabled
     */
    val isDebugMode: Boolean
        get() = prefs?.getBoolean(DEBUG_MODE_KEY, false) ?: false

    /**
     * Set debug mode
     */
    fun setDebugMode(enabled: Boolean) {
        prefs?.edit()?.putBoolean(DEBUG_MODE_KEY, enabled)?.apply()
        log("Debug mode ${if (enabled) "enabled" else "disabled"}")
    }

    /**
     * Log a debug message (only when debug mode is enabled)
     * Similar to iOS logDebug() function
     */
    fun log(message: String) {
        if (isDebugMode) {
            Log.d(TAG, message)
        }
    }

    /**
     * Log a debug message with a custom tag (only when debug mode is enabled)
     */
    fun d(tag: String, message: String) {
        if (isDebugMode) {
            Log.d(tag, message)
        }
    }

    /**
     * Log an info message with a custom tag (only when debug mode is enabled)
     */
    fun i(tag: String, message: String) {
        if (isDebugMode) {
            Log.i(tag, message)
        }
    }

    /**
     * Log a warning message with a custom tag (only when debug mode is enabled)
     */
    fun w(tag: String, message: String) {
        if (isDebugMode) {
            Log.w(tag, message)
        }
    }

    /**
     * Log an error message (always logs, regardless of debug mode)
     * Errors are always important to capture
     */
    fun e(tag: String, message: String, throwable: Throwable? = null) {
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
