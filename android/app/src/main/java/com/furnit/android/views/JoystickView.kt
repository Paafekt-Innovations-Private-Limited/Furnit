package com.furnit.android.views

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Virtual joystick view for camera control in 3D room view.
 * Matches iOS VirtualJoystick design: outer circle with draggable inner knob.
 */
class JoystickView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // Callback for joystick movement
    var onJoystickMove: ((Float, Float) -> Unit)? = null

    // Paint objects
    private val outerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#4D000000")  // Black with 30% opacity
        style = Paint.Style.FILL
    }
    private val outerStrokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#80FFFFFF")  // White with 50% opacity
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val knobPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#CCFFFFFF")  // White with 80% opacity
        style = Paint.Style.FILL
    }

    // Dimensions (matching iOS: outerCircleRadius=50, innerKnobRadius=20, maxDistance=30)
    private var outerRadius = 100f
    private var knobRadius = 40f
    private var maxDistance = 60f

    // Knob position (relative to center)
    private var knobX = 0f
    private var knobY = 0f

    // Center of the view
    private var centerX = 0f
    private var centerY = 0f

    // Dragging state
    private var isDragging = false

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        centerX = w / 2f
        centerY = h / 2f
        // Scale based on view size
        val minDimension = min(w, h).toFloat()
        outerRadius = minDimension / 2f - 4f  // Leave room for stroke
        knobRadius = outerRadius * 0.4f
        maxDistance = outerRadius * 0.6f
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // Draw outer circle background
        canvas.drawCircle(centerX, centerY, outerRadius, outerPaint)
        // Draw outer circle stroke
        canvas.drawCircle(centerX, centerY, outerRadius, outerStrokePaint)

        // Draw inner knob at offset position
        val knobScale = if (isDragging) 1.2f else 1.0f
        canvas.drawCircle(centerX + knobX, centerY + knobY, knobRadius * knobScale, knobPaint)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                isDragging = true
                updateKnobPosition(event.x, event.y)
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                updateKnobPosition(event.x, event.y)
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                isDragging = false
                // Reset knob to center
                knobX = 0f
                knobY = 0f
                invalidate()
                onJoystickMove?.invoke(0f, 0f)
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    private fun updateKnobPosition(touchX: Float, touchY: Float) {
        // Calculate offset from center
        val dx = touchX - centerX
        val dy = touchY - centerY
        val distance = sqrt(dx * dx + dy * dy)

        if (distance <= maxDistance) {
            // Within bounds - use actual position
            knobX = dx
            knobY = dy
        } else {
            // Outside bounds - constrain to max distance
            val scale = maxDistance / distance
            knobX = dx * scale
            knobY = dy * scale
        }

        invalidate()

        // Notify listener with normalized values (-1 to 1)
        val normalizedX = knobX / maxDistance
        val normalizedY = knobY / maxDistance
        onJoystickMove?.invoke(normalizedX, normalizedY)
    }
}
