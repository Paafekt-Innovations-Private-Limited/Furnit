package com.furnit.android.utils

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Human-readable default room titles for metadata and the save dialog.
 * Includes **year and time** so same-day / post-restart rooms stay distinguishable.
 */
object RoomDisplayName {
    private fun formattedSuffix(date: Date): String {
        val df = SimpleDateFormat("MMM d, yyyy, HH:mm", Locale.getDefault())
        return df.format(date)
    }

    /** Default SHARP room label (preview + SharpService metadata). */
    fun aiRoomWithTimestamp(date: Date = Date()): String = "AI Room ${formattedSuffix(date)}"

    /** Manual / photogrammetry-style room label (texture-only reconstructor, GLB save dialogs). */
    fun myRoomWithTimestamp(date: Date = Date()): String = "My Room ${formattedSuffix(date)}"
}
