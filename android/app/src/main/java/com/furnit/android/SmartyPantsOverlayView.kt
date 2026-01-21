package com.furnit.android

import android.content.Context
import android.graphics.*
import android.util.Log
import android.view.View

class SmartyPantsOverlayView(context: Context) : View(context) {
    private var maskBitmap: Bitmap? = null
    private val paint = Paint().apply { isAntiAlias = true }
    private val colorFilter = PorterDuffColorFilter(0x8000FF00.toInt(), PorterDuff.Mode.SRC_ATOP)

    fun setMask(b: Bitmap?) {
        Log.d("SmartyPantsOverlay", "setMask called: ${b?.width}x${b?.height}")
        maskBitmap = b
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        maskBitmap?.let { bmp ->
            Log.d("SmartyPantsOverlay", "Drawing mask ${bmp.width}x${bmp.height} on canvas ${width}x${height}")
            // Scale mask to view size
            val scaled = Bitmap.createScaledBitmap(bmp, width, height, true)
            // Draw with green tint overlay
            paint.alpha = 150
            paint.colorFilter = colorFilter
            canvas.drawBitmap(scaled, 0f, 0f, paint)
            paint.colorFilter = null
        }
    }
}
