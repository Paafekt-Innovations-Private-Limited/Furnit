package com.furnit.android.services

import android.os.Handler
import android.os.Looper

/**
 * Lets the home screen show room-generation progress after the user leaves
 * SinglePhotoRoomActivity while generation continues.
 */
object RoomGenerationUiState {

    private val mainHandler = Handler(Looper.getMainLooper())

    @Volatile
    var isGenerating: Boolean = false
        private set

    @Volatile
    var progress01: Float = 0f
        private set

    /** Short line for the bottom bar, e.g. "Working on it... - 42%" */
    @Volatile
    var statusLine: String = ""
        private set

    private var listener: (() -> Unit)? = null

    fun setListener(l: (() -> Unit)?) {
        listener = l
        l?.invoke()
    }

    fun update(isActive: Boolean, progress: Float, line: String) {
        isGenerating = isActive
        progress01 = progress.coerceIn(0f, 1f)
        statusLine = line
        mainHandler.post { listener?.invoke() }
    }

    fun clear() {
        update(false, 0f, "")
    }
}
