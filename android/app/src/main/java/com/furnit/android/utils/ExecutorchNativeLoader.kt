package com.furnit.android.utils

import android.util.Log

/**
 * Loads ExecuTorch native libs in an order that lets backends register (Vulkan uses static init in libexecutorch.so).
 * Call before any [org.pytorch.executorch.Module] use. Safe to call multiple times.
 */
object ExecutorchNativeLoader {

    private const val TAG = "ExecuTorchLoad"
    @Volatile
    private var loaded = false

    /**
     * Load libexecutorch_core.so (if split build), then libexecutorch.so, then libexecutorch_jni.so.
     * Missing optional libs are ignored; JNI must succeed for Java Module API.
     */
    @JvmStatic
    fun loadForJavaModule() {
        synchronized(this) {
            if (loaded) return
            // Split AAR: core first so symbols resolve; then runtime (Vulkan backend registers here); then JNI (initHybrid).
            try {
                System.loadLibrary("executorch_core")
                Log.d(TAG, "executorch_core loaded OK")
            } catch (e: UnsatisfiedLinkError) {
                Log.d(TAG, "executorch_core skipped: ${e.message}")
            }
            var executorchLoaded = false
            try {
                System.loadLibrary("executorch")
                Log.d(TAG, "executorch loaded OK (backend registration runs in this .so)")
                executorchLoaded = true
            } catch (e: UnsatisfiedLinkError) {
                Log.w(TAG, "executorch preload failed: ${e.message}")
            }
            try {
                System.loadLibrary("executorch_jni")
                Log.d(TAG, "executorch_jni loaded OK")
            } catch (e: UnsatisfiedLinkError) {
                if (!executorchLoaded) throw e
                Log.d(TAG, "executorch_jni skipped: ${e.message} (JNI is provided by libexecutorch.so in this build)")
            }
            loaded = true
        }
    }
}
