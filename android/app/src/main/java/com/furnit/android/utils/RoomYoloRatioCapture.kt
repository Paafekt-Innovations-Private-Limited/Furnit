package com.furnit.android.utils

import android.content.Context
import com.furnit.android.services.FurnitureFitManager
import java.io.File

/**
 * One-shot YOLOe on the room reference image when the room is opened; persists dimensionless
 * height fractions to [RoomFolderMetadata] for live brain segmentation ROI (see plan: ratio-based fitment).
 */
object RoomYoloRatioCapture {

    private const val TAG = "RoomYoloRatioCapture"

    fun captureIfMissing(context: Context, roomFolder: File) {
        if (!roomFolder.isDirectory) return
        val existing = RoomFolderMetadata.readFromFolder(roomFolder) ?: return
        if (existing.yoloRefImageHeightPx != null) return

        val ref = YoloRatioCalibration.pickReferenceImageFile(roomFolder) ?: run {
            LogUtil.d(TAG, "No reference image in ${roomFolder.name}")
            return
        }
        val bitmap = YoloRatioCalibration.decodeBitmapReference(ref) ?: return

        val manager = FurnitureFitManager(context.applicationContext)
        if (!manager.initializeAuto()) {
            LogUtil.w(TAG, "YOLO backend not available for ratio capture")
            manager.close()
            return
        }

        val boxes = try {
            manager.detectCalibrationBoxesSync(bitmap)
        } finally {
            manager.close()
        }

        if (boxes.isEmpty()) {
            LogUtil.d(TAG, "No detections for ratio capture in ${roomFolder.name}")
            return
        }

        val wallFrac = YoloRatioCalibration.wallHeightFractionOrFullFrame(bitmap.width, bitmap.height, boxes)
        val furn = YoloRatioCalibration.furnitureHeightFractionsByLabel(bitmap.height, boxes)
        val merged = existing.copy(
            yoloWallHeightFrac = wallFrac,
            yoloFurnitureHeightFracByClass = furn,
            yoloRefImageHeightPx = bitmap.height,
            sharpNavarroRoomHeightAtYoloCapture = existing.roomHeight,
        )
        RoomFolderMetadata.writeToFolder(roomFolder, merged)
        LogUtil.d(TAG, "Stored YOLO ratios for ${roomFolder.name} wallFrac=$wallFrac furnitureClasses=${furn.size}")
    }
}
