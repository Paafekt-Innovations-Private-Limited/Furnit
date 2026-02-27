package com.furnit.android

import android.content.Context
import android.graphics.*
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import com.furnit.android.utils.DebugLogger
import kotlin.math.max
import kotlin.math.min

// Detection data for overlay display
data class DetectionResult(
    val x: Float,      // center x in input coords (640x640)
    val y: Float,      // center y in input coords
    val w: Float,      // width in input coords
    val h: Float,      // height in input coords
    val confidence: Float,
    val label: String
)

class FurnitureFitOverlayView(context: Context) : View(context) {
    private var maskBitmap: Bitmap? = null
    private var detections: List<DetectionResult> = emptyList()
    private var inputSize = 640 // Model input size

    // Pinch-to-zoom scale factor for furniture
    private var furnitureScale = 0.6f  // Start smaller to fit in room
    private var translateX = 0f
    private var translateY = 0f

    // For single-finger drag
    private var lastTouchX = 0f
    private var lastTouchY = 0f
    private var isDraggingFurniture = false
    private var touchOnFurniture = false

    // Callback for when touch is outside furniture (for camera control)
    var onTouchOutsideFurniture: ((MotionEvent) -> Unit)? = null

    private val scaleGestureDetector = ScaleGestureDetector(context, ScaleListener())

    private val maskPaint = Paint().apply {
        isAntiAlias = true
        isFilterBitmap = true
    }

    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }

    private val textBgPaint = Paint().apply {
        color = Color.argb(200, 0, 0, 0)
        style = Paint.Style.FILL
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 36f
        isAntiAlias = true
        typeface = Typeface.DEFAULT_BOLD
    }

    private inner class ScaleListener : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector: ScaleGestureDetector): Boolean {
            furnitureScale *= detector.scaleFactor
            furnitureScale = max(0.3f, min(furnitureScale, 3.0f))  // Limit scale 0.3x to 3x
            invalidate()
            return true
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        return handleTouchInternal(event)
    }

    /**
     * Handle touch events passed from parent view (for pinch-to-zoom and drag)
     */
    fun handleExternalTouchEvent(event: MotionEvent): Boolean {
        return handleTouchInternal(event)
    }

    /**
     * True if all active pointers are inside the furniture bbox (so pinch/scale applies only to furniture).
     */
    private fun allPointersOnFurniture(event: MotionEvent): Boolean {
        for (i in 0 until event.pointerCount) {
            if (!isTouchOnFurniture(event.getX(i), event.getY(i))) return false
        }
        return event.pointerCount > 0
    }

    private fun handleTouchInternal(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                lastTouchX = event.x
                lastTouchY = event.y
                touchOnFurniture = isTouchOnFurniture(event.x, event.y)
                isDraggingFurniture = touchOnFurniture
                if (!touchOnFurniture) {
                    onTouchOutsideFurniture?.invoke(event)
                    return false
                }
                scaleGestureDetector.onTouchEvent(event)
            }
            MotionEvent.ACTION_POINTER_DOWN -> {
                if (event.pointerCount == 2) {
                    if (!allPointersOnFurniture(event)) {
                        touchOnFurniture = false
                        isDraggingFurniture = false
                        onTouchOutsideFurniture?.invoke(event)
                        return false
                    }
                    isDraggingFurniture = false
                }
                if (touchOnFurniture) {
                    scaleGestureDetector.onTouchEvent(event)
                }
            }
            MotionEvent.ACTION_MOVE -> {
                if (!touchOnFurniture) {
                    onTouchOutsideFurniture?.invoke(event)
                    return false
                }
                scaleGestureDetector.onTouchEvent(event)
                if (!scaleGestureDetector.isInProgress && event.pointerCount == 1 && isDraggingFurniture) {
                    val deltaX = event.x - lastTouchX
                    val deltaY = event.y - lastTouchY
                    translateX += deltaX
                    translateY += deltaY
                    lastTouchX = event.x
                    lastTouchY = event.y
                    invalidate()
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                if (!touchOnFurniture) {
                    onTouchOutsideFurniture?.invoke(event)
                    return false
                }
                scaleGestureDetector.onTouchEvent(event)
                val wasHandling = touchOnFurniture
                isDraggingFurniture = false
                touchOnFurniture = false
                return wasHandling
            }
            else -> if (touchOnFurniture) scaleGestureDetector.onTouchEvent(event)
        }
        return touchOnFurniture
    }

    /**
     * Check if the touch point is on the furniture (on a non-transparent pixel)
     */
    private fun isTouchOnFurniture(touchX: Float, touchY: Float): Boolean {
        val bmp = maskBitmap
        if (bmp == null) return false
        if (width == 0 || height == 0) return false

        // Same transform as onDraw: uniform base scale, pivot at bitmap center, screen center + drag
        val baseScale = min(width / bmp.width.toFloat(), height / bmp.height.toFloat())
        val totalScale = baseScale * furnitureScale
        val screenCenterX = width / 2f
        val screenCenterY = height * 0.35f
        val maskLeft = screenCenterX - (bmp.width / 2f) * totalScale + translateX
        val maskTop = screenCenterY - (bmp.height / 2f) * totalScale + translateY

        val bmpX = ((touchX - maskLeft) / totalScale).toInt()
        val bmpY = ((touchY - maskTop) / totalScale).toInt()

        if (bmpX < 0 || bmpX >= bmp.width || bmpY < 0 || bmpY >= bmp.height) return false
        val pixel = bmp.getPixel(bmpX, bmpY)
        return Color.alpha(pixel) > 10
    }

    fun setMask(b: Bitmap?) {
        maskBitmap = b
        invalidate()
    }

    fun setDetections(dets: List<DetectionResult>, modelInputSize: Int = 640) {
        detections = dets
        inputSize = modelInputSize
        invalidate()
    }

    fun setMaskAndDetections(mask: Bitmap?, dets: List<DetectionResult>, modelInputSize: Int = 640) {
        maskBitmap = mask
        detections = dets
        inputSize = modelInputSize
        invalidate()
    }

    fun resetTransform() {
        furnitureScale = 1.0f
        translateX = 0f
        translateY = 0f
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // Draw segmented objects (cutout with transparent background).
        // Ultralytics-style: uniform base scale to fit, scale around bitmap center (pivot), then translate to screen center.
        maskBitmap?.let { bmp ->
            val baseScale = min(width / bmp.width.toFloat(), height / bmp.height.toFloat())
            val totalScale = baseScale * furnitureScale

            val matrix = Matrix()
            matrix.reset()
            matrix.postScale(totalScale, totalScale, bmp.width / 2f, bmp.height / 2f)
            val screenCenterX = width / 2f
            val screenCenterY = height * 0.35f
            matrix.postTranslate(screenCenterX - bmp.width / 2f, screenCenterY - bmp.height / 2f)
            matrix.postTranslate(translateX, translateY)

            canvas.drawBitmap(bmp, matrix, maskPaint)
        }

        // Draw bounding boxes and labels only when debug mode is enabled (same screen center as mask)
        if (detections.isNotEmpty() && DebugLogger.isDebugMode) {
            val baseScale = min(width / inputSize.toFloat(), height / inputSize.toFloat())
            val totalScale = baseScale * furnitureScale
            val screenCenterX = width / 2f
            val screenCenterY = height * 0.35f
            val centerOffsetX = screenCenterX - (inputSize / 2f) * totalScale + translateX
            val centerOffsetY = screenCenterY - (inputSize / 2f) * totalScale + translateY

            for (det in detections) {
                val left = (det.x - det.w / 2) * totalScale + centerOffsetX
                val top = (det.y - det.h / 2) * totalScale + centerOffsetY
                val right = (det.x + det.w / 2) * totalScale + centerOffsetX
                val bottom = (det.y + det.h / 2) * totalScale + centerOffsetY

                // Draw bounding box
                canvas.drawRect(left, top, right, bottom, boxPaint)

                // Prepare label text
                val label = "${det.label} ${String.format("%.0f%%", det.confidence * 100)}"
                val textWidth = textPaint.measureText(label)
                val textHeight = textPaint.textSize

                // Draw label background
                val bgLeft = left
                val bgTop = top - textHeight - 8
                val bgRight = left + textWidth + 16
                val bgBottom = top
                canvas.drawRect(bgLeft, bgTop, bgRight, bgBottom, textBgPaint)

                // Draw label text
                canvas.drawText(label, left + 8, top - 8, textPaint)
            }
        }
    }
}
