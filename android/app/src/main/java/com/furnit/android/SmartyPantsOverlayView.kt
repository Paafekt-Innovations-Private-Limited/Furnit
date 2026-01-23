package com.furnit.android

import android.content.Context
import android.graphics.*
import android.util.Log
import android.view.View

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

    private val bgPaint = Paint().apply {
        color = Color.argb(200, 30, 30, 30)  // Dark semi-transparent background
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // Draw dark background where objects will be shown
        maskBitmap?.let {
            canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), bgPaint)
        }

        // Draw segmented objects (cutout with transparent background)
        maskBitmap?.let { bmp ->
            val scaleX = width.toFloat() / bmp.width
            val scaleY = height.toFloat() / bmp.height
            val matrix = Matrix()
            matrix.setScale(scaleX, scaleY)
            canvas.drawBitmap(bmp, matrix, maskPaint)
        }

        // Draw bounding boxes and labels
        if (detections.isNotEmpty()) {
            // Scale from model input coords to view coords
            val scaleX = width.toFloat() / inputSize
            val scaleY = height.toFloat() / inputSize

            for (det in detections) {
                // Convert center coords to corner coords and scale to view
                val left = (det.x - det.w / 2) * scaleX
                val top = (det.y - det.h / 2) * scaleY
                val right = (det.x + det.w / 2) * scaleX
                val bottom = (det.y + det.h / 2) * scaleY

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
