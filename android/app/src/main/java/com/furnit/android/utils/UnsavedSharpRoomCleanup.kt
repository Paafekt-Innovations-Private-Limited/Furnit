package com.furnit.android.utils

import android.content.Context
import java.io.File

/**
 * Removes SHARP folders under `files/sharp_rooms/` that were never committed via Save ([RoomFolderMetadata.Snapshot.previewOnly]).
 * Process death skips [com.furnit.android.SharpRoomActivity] `onDestroy` cleanup; this runs at app cold start.
 */
object UnsavedSharpRoomCleanup {

    private const val SHARP_ROOMS_DIR = "sharp_rooms"

    fun deletePreviewOnlyFolders(context: Context) {
        val root = File(context.filesDir, SHARP_ROOMS_DIR)
        if (!root.isDirectory) return
        val children = root.listFiles() ?: return
        for (folder in children) {
            if (!folder.isDirectory) continue
            val snap = RoomFolderMetadata.readFromFolder(folder) ?: continue
            if (snap.previewOnly == true) {
                try {
                    val ok = folder.deleteRecursively()
                    LogUtil.d("UnsavedSharpRoomCleanup", "Removed preview-only SHARP folder ${folder.name} ok=$ok")
                } catch (e: Exception) {
                    LogUtil.w("UnsavedSharpRoomCleanup", "Failed to delete ${folder.absolutePath}", e)
                }
            }
        }
    }
}
