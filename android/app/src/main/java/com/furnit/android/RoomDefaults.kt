package com.furnit.android

import android.content.Context
import android.content.SharedPreferences

object RoomDefaults {
    private const val PREFS_NAME = "furnit_prefs"
    private const val KEY_ROOM_WIDTH_M = "default_room_width_m"
    private const val KEY_ROOM_HEIGHT_M = "default_room_height_m"
    private const val KEY_ROOM_DEPTH_M = "default_room_depth_m"

    const val DEFAULT_WIDTH_M = 4.0f
    const val DEFAULT_HEIGHT_M = 3.0f
    const val DEFAULT_DEPTH_M = 4.5f

    fun prefs(context: Context): SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun widthMeters(prefs: SharedPreferences): Float =
        sanitize(prefs.getFloat(KEY_ROOM_WIDTH_M, DEFAULT_WIDTH_M), DEFAULT_WIDTH_M, 2.0f, 10.0f)

    fun heightMeters(prefs: SharedPreferences): Float =
        sanitize(prefs.getFloat(KEY_ROOM_HEIGHT_M, DEFAULT_HEIGHT_M), DEFAULT_HEIGHT_M, 2.0f, 5.0f)

    fun depthMeters(prefs: SharedPreferences): Float =
        sanitize(prefs.getFloat(KEY_ROOM_DEPTH_M, DEFAULT_DEPTH_M), DEFAULT_DEPTH_M, 2.0f, 10.0f)

    fun widthMeters(context: Context): Float = widthMeters(prefs(context))

    fun heightMeters(context: Context): Float = heightMeters(prefs(context))

    fun depthMeters(context: Context): Float = depthMeters(prefs(context))

    fun setWidthMeters(prefs: SharedPreferences, value: Float) {
        prefs.edit().putFloat(KEY_ROOM_WIDTH_M, sanitize(value, DEFAULT_WIDTH_M, 2.0f, 10.0f)).apply()
    }

    fun setHeightMeters(prefs: SharedPreferences, value: Float) {
        prefs.edit().putFloat(KEY_ROOM_HEIGHT_M, sanitize(value, DEFAULT_HEIGHT_M, 2.0f, 5.0f)).apply()
    }

    fun setDepthMeters(prefs: SharedPreferences, value: Float) {
        prefs.edit().putFloat(KEY_ROOM_DEPTH_M, sanitize(value, DEFAULT_DEPTH_M, 2.0f, 10.0f)).apply()
    }

    private fun sanitize(value: Float, fallback: Float, min: Float, max: Float): Float {
        if (!value.isFinite()) return fallback
        return value.coerceIn(min, max)
    }
}
