package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import com.furnit.android.utils.LogUtil

/**
 * Shared still-image YOLO-E (NCNN) path for **wall measurement** and any code that must mirror
 * [FurnitureFitManager] without furniture-only filtering.
 *
 * iOS parity: [FurnitureFitView] uses `YoloEImageInference` + empty `classBlacklist`; Android has no
 * `blacklist.json` in Kotlin — this API never applies one (wall/door classes must stay visible).
 */
object YoloEImageInference {

    private const val TAG = "WALL_MEAS"

    /** Furniture Fit / AR path default; wall measurement uses [WALL_MEAS_CONF] instead. */
    const val CONF_THRESHOLD: Float = 0.25f

    /** Default for wall measurement when caller does not pass a floor (prefer [WallMeasurementEstimator] explicit 0.05). */
    const val WALL_MEAS_CONF: Float = 0.05f

    const val IOU_THRESHOLD: Float = 0.45f

    /**
     * Runs `yoloe-11l-seg` NCNN on [bitmap] with [confThreshold] (use [WALL_MEAS_CONF] for wall measurement).
     * Does **not** filter by any furniture blacklist (none is applied on Android today; kept explicit for parity with iOS).
     */
    fun runDetectionsUnfiltered(context: Context, bitmap: Bitmap, confThreshold: Float = WALL_MEAS_CONF): List<NcnnYoloe.Detection> {
        if (!NcnnYoloe.isAvailable()) {
            LogUtil.w(TAG, "yolo NCNN library not loaded")
            return emptyList()
        }
        val yolo = NcnnYoloe()
        if (!yolo.init(context.applicationContext)) {
            LogUtil.w(TAG, "yolo NCNN init failed (same assets as FurnitureFit: yoloe-11l-seg)")
            return emptyList()
        }
        return try {
            LogUtil.i(
                TAG,
                "yolo pipeline=FurnitureFit NCNN conf=$confThreshold iou=$IOU_THRESHOLD " +
                    "model_side=${NcnnYoloe.MODEL_INPUT_SIDE} classBlacklist=none (wall_meas: anchor class score floor)",
            )
            yolo.detect(bitmap, confThreshold, IOU_THRESHOLD)
        } finally {
            yolo.release()
        }
    }
}
