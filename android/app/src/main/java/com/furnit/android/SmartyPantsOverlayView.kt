package com.furnit.android

import android.content.Context
import android.graphics.*
import android.view.View

class SmartyPantsOverlayView(context: Context) : View(context) {
    private var maskBitmap: Bitmap? = null
    private val paint = Paint().apply { isAntiAlias = true }

    fun setMask(b: Bitmap?) {
        maskBitmap = b
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        maskBitmap?.let { bmp ->
            // Draw mask tinted
            val scaled = Bitmap.createScaledBitmap(bmp, width, height, true)
            paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.DST_OVER)
            paint.alpha = 180
            canvas.drawBitmap(scaled, 0f, 0f, paint)
            paint.xfermode = null
        }
    }
}
