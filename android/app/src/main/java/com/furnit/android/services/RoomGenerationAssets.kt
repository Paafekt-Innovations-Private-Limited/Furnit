package com.furnit.android.services

import android.content.Context
import com.furnit.android.utils.LogUtil
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

object RoomGenerationAssets {
    private const val TAG = "RoomGenerationAssets"

    const val DEPTH_ANYTHING_METRIC_INDOOR_SMALL_ONNX =
        "room_generation/depth_anything/depth_anything_v2_metric_indoor_small.onnx"
    const val GEOCALIB_PINHOLE_CNN_ONNX =
        "room_generation/geocalib/geocalib_pinhole_cnn.onnx"
    const val RTMDET_INS_M_RAW_ONNX = "rtmdet-ins-m-raw.onnx"

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
            id = "rtmdet_ins_m",
            assetPath = RTMDET_INS_M_RAW_ONNX,
            role = "object_masking",
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
    ) {
        val hasSwiftParityAssets: Boolean
            get() = missing.isEmpty()
    }

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

    @Throws(IOException::class)
    fun copyAssetToCache(context: Context, assetPath: String): File {
        val outFile = File(context.cacheDir, assetPath)
        outFile.parentFile?.mkdirs()
        context.assets.open(assetPath).use { input ->
            FileOutputStream(outFile).use { output ->
                input.copyTo(output)
            }
        }
        return outFile
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
