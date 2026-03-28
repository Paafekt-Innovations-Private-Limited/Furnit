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

    // Pinch-to-zoom scale factor for furniture (1.0 = neutral)
    private var furnitureScale = 1.0f
    /** AR-assisted overlay scale from the host (1f when AR is off or not yet valid). */
    private var assistedOverlayScale = 1f
    private var displayedFurnitureHeightMeters: Float? = null
    private var roomHeightMeters: Float? = null
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
     * Check if the touch point is on the furniture: use detection bbox when available (furniture structure),
     * otherwise fall back to mask pixel (non-transparent).
     */
    private fun isTouchOnFurniture(touchX: Float, touchY: Float): Boolean {
        if (width == 0 || height == 0) return false
        val screenCenterX = width / 2f
        val screenCenterY = overlayScreenCenterY()

        // Prefer detection bbox (furniture structure) for hit-test when we have detections
        if (detections.isNotEmpty() && inputSize > 0) {
            val baseScale = min(width / inputSize.toFloat(), height / inputSize.toFloat())
            val totalScaleX = baseScale * furnitureScale * assistedOverlayScale
            val totalScaleY = totalScaleX * computeVerticalClampFactor(totalScaleX)
            val centerOffsetX = screenCenterX - (inputSize / 2f) * totalScaleX + translateX
            val centerOffsetY = screenCenterY - (inputSize / 2f) * totalScaleY + translateY
            for (det in detections) {
                val left = (det.x - det.w / 2) * totalScaleX + centerOffsetX
                val top = (det.y - det.h / 2) * totalScaleY + centerOffsetY
                val right = (det.x + det.w / 2) * totalScaleX + centerOffsetX
                val bottom = (det.y + det.h / 2) * totalScaleY + centerOffsetY
                if (touchX in left..right && touchY in top..bottom) return true
            }
            return false
        }

        // Fallback: mask pixel (segmented shape)
        val bmp = maskBitmap ?: return false
        val baseScale = min(width / bmp.width.toFloat(), height / bmp.height.toFloat())
        val totalScaleX = baseScale * furnitureScale * assistedOverlayScale
        val totalScaleY = totalScaleX * computeVerticalClampFactor(totalScaleX)
        val maskLeft = screenCenterX - (bmp.width / 2f) * totalScaleX + translateX
        val maskTop = screenCenterY - (bmp.height / 2f) * totalScaleY + translateY
        val bmpX = ((touchX - maskLeft) / totalScaleX).toInt()
        val bmpY = ((touchY - maskTop) / totalScaleY).toInt()
        if (bmpX < 0 || bmpX >= bmp.width || bmpY < 0 || bmpY >= bmp.height) return false
        return Color.alpha(bmp.getPixel(bmpX, bmpY)) > 10
    }

    private fun computeVerticalClampFactor(totalScaleX: Float): Float {
        val displayedHeight = displayedFurnitureHeightMeters
        val roomHeight = roomHeightMeters
        if (displayedHeight == null || roomHeight == null || displayedHeight <= roomHeight || height <= 1) return 1f
        val referenceDetectionHeight = detections.firstOrNull()?.h
        val referenceHeightPx = referenceDetectionHeight?.takeIf { it > 1f }
            ?: maskBitmap?.height?.toFloat()
            ?: return 1f
        val renderedHeightPx = referenceHeightPx * totalScaleX
        if (renderedHeightPx <= 1f) return 1f
        val maxAllowedHeightPx = height * 0.60f
        return (maxAllowedHeightPx / renderedHeightPx).coerceIn(0.1f, 1f)
    }

    private fun overlayScreenCenterY(): Float {
        val fillAnchorY = height * 0.35f
        val centeredAnchorY = height * 0.5f
        val relaxedStartScale = 1.15f
        val centeredScale = 0.75f
        if (furnitureScale >= relaxedStartScale) return fillAnchorY
        if (furnitureScale <= centeredScale) return centeredAnchorY
        val progress = ((relaxedStartScale - furnitureScale) / (relaxedStartScale - centeredScale)).coerceIn(0f, 1f)
        return fillAnchorY + (centeredAnchorY - fillAnchorY) * progress
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

    fun setMaskAndDetections(
        mask: Bitmap?,
        dets: List<DetectionResult>,
        modelInputSize: Int = 640,
        assistedScale: Float = 1f,
        displayedHeightMeters: Float? = null,
        roomHeightMeters: Float? = null,
    ) {
        maskBitmap = mask
        detections = dets
        inputSize = modelInputSize
        this.displayedFurnitureHeightMeters = displayedHeightMeters
        this.roomHeightMeters = roomHeightMeters
        assistedOverlayScale = if (mask == null) {
            1f
        } else {
            assistedScale.coerceIn(0.25f, 4f)
        }
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
            val totalScaleX = baseScale * furnitureScale * assistedOverlayScale
            val totalScaleY = totalScaleX * computeVerticalClampFactor(totalScaleX)

            val matrix = Matrix()
            matrix.reset()
            matrix.postScale(totalScaleX, totalScaleY, bmp.width / 2f, bmp.height / 2f)
            val screenCenterX = width / 2f
            val screenCenterY = overlayScreenCenterY()
            matrix.postTranslate(screenCenterX - bmp.width / 2f, screenCenterY - bmp.height / 2f)
            matrix.postTranslate(translateX, translateY)

            canvas.drawBitmap(bmp, matrix, maskPaint)
        }

        // Draw bounding boxes and labels only when debug mode is enabled (same screen center as mask)
        if (detections.isNotEmpty() && DebugLogger.isDebugMode) {
            val baseScale = min(width / inputSize.toFloat(), height / inputSize.toFloat())
            val totalScaleX = baseScale * furnitureScale * assistedOverlayScale
            val totalScaleY = totalScaleX * computeVerticalClampFactor(totalScaleX)
            val screenCenterX = width / 2f
            val screenCenterY = overlayScreenCenterY()
            val centerOffsetX = screenCenterX - (inputSize / 2f) * totalScaleX + translateX
            val centerOffsetY = screenCenterY - (inputSize / 2f) * totalScaleY + translateY

            for (det in detections) {
                val left = (det.x - det.w / 2) * totalScaleX + centerOffsetX
                val top = (det.y - det.h / 2) * totalScaleY + centerOffsetY
                val right = (det.x + det.w / 2) * totalScaleX + centerOffsetX
                val bottom = (det.y + det.h / 2) * totalScaleY + centerOffsetY

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
