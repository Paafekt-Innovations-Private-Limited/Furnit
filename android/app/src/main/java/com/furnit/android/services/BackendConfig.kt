package com.furnit.android.services

/**
 * Central feature flags for inference backends.
 *
 * Goal: keep wrappers in-tree, but allow shipping an "ONNX-only" app build
 * that never attempts to initialize or run experimental backends.
 */
object BackendConfig {
    const val ENABLE_NCNN: Boolean = true
    const val ENABLE_EXECUTORCH: Boolean = true
    const val ENABLE_LITERT: Boolean = true
    const val ENABLE_PYTHON: Boolean = true
    const val ENABLE_TORCH_MOBILE: Boolean = true
    const val ENABLE_NATIVE_PT: Boolean = true
    const val ENABLE_LITERT_GPU: Boolean = false
    const val ENABLE_LITERT_NNAPI: Boolean = false
    const val ENABLE_ONNX_INT8: Boolean = true
    const val ENABLE_ONNX_FP16: Boolean = true
    const val ENABLE_EXECUTORCH_FP16: Boolean = true
    const val ENABLE_EXECUTORCH_INT8: Boolean = true

    fun isEnabled(backendId: String): Boolean {
        return when (backendId) {
            "onnx" -> true
            "onnx_int8" -> ENABLE_ONNX_INT8
            "onnx_fp16" -> ENABLE_ONNX_FP16
            "ncnn" -> ENABLE_NCNN
            "executorch" -> ENABLE_EXECUTORCH
            "executorch_fp16" -> ENABLE_EXECUTORCH_FP16
            "executorch_int8" -> ENABLE_EXECUTORCH_INT8
            "litert" -> ENABLE_LITERT
            "python" -> ENABLE_PYTHON
            "torch_mobile" -> ENABLE_TORCH_MOBILE
            "native_pt" -> ENABLE_NATIVE_PT
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
