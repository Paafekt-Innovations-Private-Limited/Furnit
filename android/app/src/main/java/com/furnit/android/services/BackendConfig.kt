package com.furnit.android.services

/**
 * Central feature flags for inference backends.
 *
 * Goal: keep wrappers in-tree, but allow shipping an "ONNX-only" app build
 * that never attempts to initialize or run experimental backends.
 */
object BackendConfig {
    /**
     * Only ONNX Runtime is required to work and be deployed to devices.
     * Keep other backends disabled by default to avoid runtime failures and
     * to avoid requiring their model files during setup.
     */
    const val ENABLE_NCNN: Boolean = false
    const val ENABLE_EXECUTORCH: Boolean = false
    const val ENABLE_LITERT: Boolean = false

    fun isEnabled(backendId: String): Boolean {
        return when (backendId) {
            "onnx" -> true
            "ncnn" -> ENABLE_NCNN
            "executorch" -> ENABLE_EXECUTORCH
            "litert" -> ENABLE_LITERT
            else -> false
        }
    }

    /**
     * If a persisted preference points at a disabled backend, normalize to ONNX.
     */
    fun normalize(backendId: String?): String {
        val id = backendId ?: "onnx"
        return if (isEnabled(id)) id else "onnx"
    }
}

