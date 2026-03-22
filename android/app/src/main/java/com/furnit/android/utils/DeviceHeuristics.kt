package com.furnit.android.utils

import android.os.Build
import java.util.Locale

/**
 * Lightweight device classification for ML backend defaults (no heavy reflection).
 */
object DeviceHeuristics {

    /**
     * Google Pixel phones / tablets (Tensor). Full Vulkan Part1+2 is fragile here (timeouts, device lost);
     * hybrid Part1+2 on CPU + Part3/4 on Vulkan is the supported path when INT8 sidecars exist (prefer models_cpuvulkan_hybrid).
     */
    fun isGooglePixelFamily(): Boolean {
        if (!Build.MANUFACTURER.equals("Google", ignoreCase = true)) return false
        val model = Build.MODEL?.lowercase(Locale.US).orEmpty()
        val product = Build.PRODUCT?.lowercase(Locale.US).orEmpty()
        val device = Build.DEVICE?.lowercase(Locale.US).orEmpty()
        return model.contains("pixel") ||
            product.contains("pixel") ||
            device.contains("pixel") ||
            product.startsWith("redfin") ||
            product.startsWith("bramble") ||
            product.startsWith("sunfish") ||
            product.startsWith("coral") ||
            product.startsWith("flame") ||
            product.startsWith("crosshatch") ||
            product.startsWith("blueline") ||
            product.startsWith("bonito") ||
            product.startsWith("sargo") ||
            product.startsWith("barbet")
    }
}
