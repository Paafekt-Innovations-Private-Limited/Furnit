package com.furnit.android.services

import android.content.SharedPreferences

object ExecutorchFixedSettings {
    private const val KEY_USE_TRUE_1280 = "executorch_int8_use_1280"
    private const val KEY_PREFER_VULKAN_FP16 = "executorch_vulkan_prefer_fp16"
    private const val KEY_PREFER_SINGLE_PART4B = "executorch_prefer_single_part4b"

    const val USE_TRUE_1280: Boolean = false
    const val PREFER_VULKAN_FP16: Boolean = true
    const val PREFER_SINGLE_PART4B: Boolean = false

    fun syncToPrefs(prefs: SharedPreferences) {
        val editor = prefs.edit()
        var changed = false

        if (!prefs.contains(KEY_USE_TRUE_1280) || prefs.getBoolean(KEY_USE_TRUE_1280, !USE_TRUE_1280) != USE_TRUE_1280) {
            editor.putBoolean(KEY_USE_TRUE_1280, USE_TRUE_1280)
            changed = true
        }
        if (!prefs.contains(KEY_PREFER_VULKAN_FP16) || prefs.getBoolean(KEY_PREFER_VULKAN_FP16, !PREFER_VULKAN_FP16) != PREFER_VULKAN_FP16) {
            editor.putBoolean(KEY_PREFER_VULKAN_FP16, PREFER_VULKAN_FP16)
            changed = true
        }
        if (!prefs.contains(KEY_PREFER_SINGLE_PART4B) || prefs.getBoolean(KEY_PREFER_SINGLE_PART4B, !PREFER_SINGLE_PART4B) != PREFER_SINGLE_PART4B) {
            editor.putBoolean(KEY_PREFER_SINGLE_PART4B, PREFER_SINGLE_PART4B)
            changed = true
        }

        if (changed) editor.apply()
    }
}
