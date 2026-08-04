package com.furnit.android.services

import android.content.Context
import com.furnit.android.utils.LogUtil
import java.io.IOException

object RoomGenerationAssets {
    private const val TAG = "RoomGenerationAssets"

    const val DEPTH_ANYTHING_METRIC_INDOOR_SMALL_ONNX =
        "room_generation/depth_anything/depth_anything_v2_metric_indoor_small.onnx"
    const val GEOCALIB_PINHOLE_CNN_ONNX =
        "room_generation/geocalib/geocalib_pinhole_cnn.onnx"
    const val RTMDET_INS_M_RAW_FP16_TFLITE = "rtmdet-ins-m-raw-fp16.tflite"

    private val swiftParitySpecs = listOf(
        AssetSpec(
            id = "depth_anything_metric_indoor_small",
            assetPath = DEPTH_ANYTHING_METRIC_INDOOR_SMALL_ONNX,
            role = "metric_depth",
        ),
        AssetSpec(
            id = "geocalib_pinhole_cnn",
            assetPath = GEOCALIB_PINHOLE_CNN_ONNX,
            role = "camera_calibration",
        ),
        AssetSpec(
            id = "rtmdet_ins_m_fp16_litert",
            assetPath = RTMDET_INS_M_RAW_FP16_TFLITE,
            role = "accelerated_object_masking",
        ),
    )

    @Volatile private var loggedAvailability = false

    data class AssetSpec(
        val id: String,
        val assetPath: String,
        val role: String,
    )

    data class Availability(
        val present: List<AssetSpec>,
        val missing: List<AssetSpec>,
    )

    fun checkAvailability(context: Context): Availability {
        val present = mutableListOf<AssetSpec>()
        val missing = mutableListOf<AssetSpec>()
        for (spec in swiftParitySpecs) {
            if (assetExists(context, spec.assetPath)) {
                present += spec
            } else {
                missing += spec
            }
        }
        return Availability(present = present, missing = missing)
    }

    fun logAvailability(context: Context) {
        if (loggedAvailability) return
        synchronized(this) {
            if (loggedAvailability) return
            val availability = checkAvailability(context)
            val present = availability.present.joinToString { "${it.id}:${it.assetPath}" }
            LogUtil.i(TAG, "Packaged room generation assets: ${present.ifBlank { "none" }}")
            if (availability.missing.isNotEmpty()) {
                val missing = availability.missing.joinToString { "${it.id}:${it.assetPath}" }
                LogUtil.w(TAG, "Missing Swift-parity room generation assets: $missing")
            }
            loggedAvailability = true
        }
    }

    private fun assetExists(context: Context, assetPath: String): Boolean {
        return try {
            context.assets.open(assetPath).close()
            true
        } catch (_: IOException) {
            false
        }
    }
}
