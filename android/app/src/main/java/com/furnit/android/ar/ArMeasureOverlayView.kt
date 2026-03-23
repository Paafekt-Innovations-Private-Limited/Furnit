package com.furnit.android.ar

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View

/**
 * Draws 2D line and dots for the last two projected anchor positions (screen space).
 */
class ArMeasureOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0,
) : View(context, attrs, defStyleAttr) {

    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#00E5FF")
        style = Paint.Style.STROKE
        strokeWidth = 6f * resources.displayMetrics.density
        strokeCap = Paint.Cap.ROUND
    }
    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FF9800")
        style = Paint.Style.FILL
    }
    private val path = Path()

    @Volatile
    private var screenPoints: List<Pair<Float, Float>> = emptyList()

    fun setProjectedPoints(points: List<Pair<Float, Float>>) {
        screenPoints = points
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val pts = screenPoints
        if (pts.isEmpty()) return
        val r = 18f * resources.displayMetrics.density
        for ((x, y) in pts) {
            canvas.drawCircle(x, y, r, dotPaint)
        }
        if (pts.size >= 2) {
            val a = pts[pts.size - 2]
            val b = pts[pts.size - 1]
            path.reset()
            path.moveTo(a.first, a.second)
            path.lineTo(b.first, b.second)
            canvas.drawPath(path, linePaint)
        }
    }
}
