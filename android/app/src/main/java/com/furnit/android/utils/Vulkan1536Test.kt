package com.furnit.android.utils

import android.util.Log

/**
 * Vulkan tests and diagnostics: 1536 support check and full device/extensions/sync inspection.
 * - 1536 test: logcat tag "Vulkan1536Test"
 * - Full diagnostics: logcat tag "VulkanDiag" (device, extensions, sync-related; shader note).
 */
object Vulkan1536Test {
    private const val TAG = "Vulkan1536Test"
    const val VULKAN_DIAG_TAG = "VulkanDiag"

    /** ADB command to see Vulkan & ExecuTorch diagnostic output. */
    const val ADB_FILTER_DIAG = "adb logcat -s VulkanDiag:D Vulkan1536Test:D"

    @JvmStatic
    external fun runVulkan1536Test()

    @JvmStatic
    external fun runVulkanDiagnostics()

    @JvmStatic
    fun runAndLog() {
        try {
            System.loadLibrary("vulkan_1536_test")
            runVulkan1536Test()
            Log.i(TAG, "Check complete; see Vulkan1536Test lines above for 1536x1536 support")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "vulkan_1536_test not loaded: ${e.message}")
        }
    }

    /**
     * Run full Vulkan diagnostics (device name, driver, API version, device extensions,
     * sync-related: VK_KHR_synchronization2, VK_KHR_timeline_semaphore) and log ExecuTorch note.
     * Shader registry is inside ExecuTorch lib and not enumerable from the app.
     * @return The adb command to filter and see the diagnostic log.
     */
    @JvmStatic
    fun runDiagnosticsAndLog(): String {
        try {
            System.loadLibrary("vulkan_1536_test")
            runVulkanDiagnostics()
            Log.i(VULKAN_DIAG_TAG, "ExecuTorch lib path: see logcat nativeloader lines for libexecutorch.so / libsharp_executorch_full.so")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(VULKAN_DIAG_TAG, "vulkan_1536_test not loaded: ${e.message}")
        }
        return ADB_FILTER_DIAG
    }
}
