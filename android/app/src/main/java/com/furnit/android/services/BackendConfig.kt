package com.furnit.android.services

/**
 * Central feature flags for inference backends.
 *
 * This app build ships **ExecuTorch INT8 SHARP only** ([ExecutorchInt8Sharp]).
 * Legacy preference strings are normalized to `executorch_int8`.
 */
object BackendConfig {
    const val ENABLE_EXECUTORCH_INT8: Boolean = true

    fun isEnabled(backendId: String): Boolean {
        return when (backendId) {
            "executorch_int8" -> ENABLE_EXECUTORCH_INT8
            "onnx", "onnx_int8", "onnx_fp16",
            "executorch", "executorch_fp16",
            "python", "torch_mobile",
            "ncnn", "litert", "native_pt" -> false
            else -> false
        }
    }

    /**
     * If a persisted preference points at a removed/disabled backend, normalize to ExecuTorch INT8.
     */
    fun normalize(backendId: String?): String {
        val id = backendId ?: "executorch_int8"
        return if (isEnabled(id)) id else "executorch_int8"
    }
}
