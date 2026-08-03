package com.furnit.android.utils

import android.content.Context
import android.text.format.DateFormat
import com.furnit.android.R
import java.util.Date

/**
 * Human-readable default room titles for metadata and the save dialog.
 * Includes **year and time** so same-day / post-restart rooms stay distinguishable.
 */
object RoomDisplayName {
    private fun formattedSuffix(context: Context, date: Date): String {
        val dateText = DateFormat.getMediumDateFormat(context).format(date)
        val timeText = DateFormat.getTimeFormat(context).format(date)
        return "$dateText $timeText"
    }

    /** Default AI room label. */
    fun aiRoomWithTimestamp(context: Context, date: Date = Date()): String =
        context.getString(R.string.room_default_ai_name, formattedSuffix(context, date))

    /** Manual / photogrammetry-style room label (texture-only reconstructor, GLB save dialogs). */
    fun myRoomWithTimestamp(context: Context, date: Date = Date()): String =
        context.getString(R.string.room_default_my_name, formattedSuffix(context, date))
}
