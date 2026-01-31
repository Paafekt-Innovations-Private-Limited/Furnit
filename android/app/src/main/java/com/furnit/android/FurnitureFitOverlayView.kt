package com.furnit.android

import android.content.Context
import android.graphics.*
import android.util.Log
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

    private fun handleTouchInternal(event: MotionEvent): Boolean {
        // Always let scale gesture detector process the event for pinch-to-zoom
        scaleGestureDetector.onTouchEvent(event)

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                lastTouchX = event.x
                lastTouchY = event.y
                // Check if touch is on the furniture mask
                touchOnFurniture = isTouchOnFurniture(event.x, event.y)
                isDraggingFurniture = touchOnFurniture
                Log.d("FurnitureOverlay", "ACTION_DOWN at (${event.x}, ${event.y}) - onFurniture=$touchOnFurniture, hasMask=${maskBitmap != null}")

                // If touch is outside furniture, notify for camera control
                if (!touchOnFurniture) {
                    Log.d("FurnitureOverlay", "Invoking onTouchOutsideFurniture callback")
                    onTouchOutsideFurniture?.invoke(event)
                }
            }
            MotionEvent.ACTION_MOVE -> {
                if (touchOnFurniture) {
                    // Drag furniture if touch started on it
                    if (!scaleGestureDetector.isInProgress && event.pointerCount == 1 && isDraggingFurniture) {
                        val deltaX = event.x - lastTouchX
                        val deltaY = event.y - lastTouchY
                        translateX += deltaX
                        translateY += deltaY
                        lastTouchX = event.x
                        lastTouchY = event.y
                        invalidate()
                    }
                } else {
                    // Touch outside furniture - pass to camera control
                    onTouchOutsideFurniture?.invoke(event)
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                if (!touchOnFurniture) {
                    onTouchOutsideFurniture?.invoke(event)
                }
                isDraggingFurniture = false
                touchOnFurniture = false
            }
            MotionEvent.ACTION_POINTER_DOWN -> {
                // Second finger down - stop dragging, let pinch take over
                isDraggingFurniture = false
            }
        }
        return true
    }

    /**
     * Check if the touch point is on the furniture (on a non-transparent pixel)
     */
    private fun isTouchOnFurniture(touchX: Float, touchY: Float): Boolean {
        val bmp = maskBitmap
        if (bmp == null) {
            Log.d("FurnitureOverlay", "isTouchOnFurniture: no mask, returning false")
            return false
        }
        if (width == 0 || height == 0) {
            Log.d("FurnitureOverlay", "isTouchOnFurniture: view size is 0")
            return false
        }

        // Calculate mask transform (same as onDraw for mask)
        val baseScaleX = width.toFloat() / bmp.width
        val baseScaleY = height.toFloat() / bmp.height

        val scaledWidth = bmp.width * baseScaleX * furnitureScale
        val scaledHeight = bmp.height * baseScaleY * furnitureScale
        val centerOffsetX = (width - scaledWidth) / 2
        val floorOffsetY = height * 0.35f

        val maskLeft = centerOffsetX + translateX
        val maskTop = floorOffsetY + translateY

        // Convert touch coordinates to bitmap pixel coordinates
        val bmpX = ((touchX - maskLeft) / (baseScaleX * furnitureScale)).toInt()
        val bmpY = ((touchY - maskTop) / (baseScaleY * furnitureScale)).toInt()

        // Check if within bitmap bounds
        if (bmpX < 0 || bmpX >= bmp.width || bmpY < 0 || bmpY >= bmp.height) {
            Log.d("FurnitureOverlay", "isTouchOnFurniture: touch outside bitmap bounds, bmpCoord=($bmpX,$bmpY)")
            return false
        }

        // Check if the pixel at this location is non-transparent (alpha > 0)
        val pixel = bmp.getPixel(bmpX, bmpY)
        val alpha = Color.alpha(pixel)
        val isOnFurniture = alpha > 10  // Small threshold for anti-aliased edges

        Log.d("FurnitureOverlay", "isTouchOnFurniture: touch=($touchX,$touchY) bmpCoord=($bmpX,$bmpY) alpha=$alpha isOnFurniture=$isOnFurniture")

        return isOnFurniture
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

        // Draw segmented objects (cutout with transparent background)
        maskBitmap?.let { bmp ->
            val baseScaleX = width.toFloat() / bmp.width
            val baseScaleY = height.toFloat() / bmp.height

            val matrix = Matrix()

            // Apply base scale to fit view
            matrix.setScale(baseScaleX * furnitureScale, baseScaleY * furnitureScale)

            // Position furniture on the floor (lower part of room)
            val scaledWidth = bmp.width * baseScaleX * furnitureScale
            val scaledHeight = bmp.height * baseScaleY * furnitureScale
            val centerOffsetX = (width - scaledWidth) / 2
            // Place furniture on floor area (bottom 40% of screen)
            val floorOffsetY = height * 0.35f  // Move down to floor level

            // Apply translation (floor offset + user drag)
            matrix.postTranslate(centerOffsetX + translateX, floorOffsetY + translateY)

            canvas.drawBitmap(bmp, matrix, maskPaint)
        }

        // Draw bounding boxes and labels only when debug mode is enabled
        if (detections.isNotEmpty() && DebugLogger.isDebugMode) {
            val baseScaleX = width.toFloat() / inputSize
            val baseScaleY = height.toFloat() / inputSize

            val scaledWidth = width * furnitureScale
            val floorOffsetY = height * 0.35f
            val centerOffsetX = (width - scaledWidth) / 2 + translateX
            val centerOffsetY = floorOffsetY + translateY

            for (det in detections) {
                // Convert center coords to corner coords and scale to view
                val left = (det.x - det.w / 2) * baseScaleX * furnitureScale + centerOffsetX
                val top = (det.y - det.h / 2) * baseScaleY * furnitureScale + centerOffsetY
                val right = (det.x + det.w / 2) * baseScaleX * furnitureScale + centerOffsetX
                val bottom = (det.y + det.h / 2) * baseScaleY * furnitureScale + centerOffsetY

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
