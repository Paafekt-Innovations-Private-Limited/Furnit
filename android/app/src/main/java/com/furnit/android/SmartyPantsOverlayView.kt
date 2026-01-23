package com.furnit.android

import android.content.Context
import android.graphics.*
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
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

class SmartyPantsOverlayView(context: Context) : View(context) {
    private var maskBitmap: Bitmap? = null
    private var detections: List<DetectionResult> = emptyList()
    private var inputSize = 640 // Model input size

    // Pinch-to-zoom scale factor for furniture
    private var furnitureScale = 1.0f
    private var translateX = 0f
    private var translateY = 0f
    private var lastTouchX = 0f
    private var lastTouchY = 0f

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
        scaleGestureDetector.onTouchEvent(event)

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                lastTouchX = event.x
                lastTouchY = event.y
            }
            MotionEvent.ACTION_MOVE -> {
                if (!scaleGestureDetector.isInProgress && event.pointerCount == 1) {
                    // Single finger drag to move furniture
                    translateX += event.x - lastTouchX
                    translateY += event.y - lastTouchY
                    lastTouchX = event.x
                    lastTouchY = event.y
                    invalidate()
                }
            }
        }
        return true
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

            // Center the scaled furniture
            val scaledWidth = bmp.width * baseScaleX * furnitureScale
            val scaledHeight = bmp.height * baseScaleY * furnitureScale
            val centerOffsetX = (width - scaledWidth) / 2
            val centerOffsetY = (height - scaledHeight) / 2

            // Apply translation (center offset + user drag)
            matrix.postTranslate(centerOffsetX + translateX, centerOffsetY + translateY)

            canvas.drawBitmap(bmp, matrix, maskPaint)
        }

        // Draw bounding boxes and labels (scaled with furniture)
        if (detections.isNotEmpty()) {
            val baseScaleX = width.toFloat() / inputSize
            val baseScaleY = height.toFloat() / inputSize

            val scaledWidth = width * furnitureScale
            val scaledHeight = height * furnitureScale
            val centerOffsetX = (width - scaledWidth) / 2 + translateX
            val centerOffsetY = (height - scaledHeight) / 2 + translateY

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
