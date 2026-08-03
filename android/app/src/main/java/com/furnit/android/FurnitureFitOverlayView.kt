package com.furnit.android

import android.content.Context
import android.graphics.*
import android.os.Build
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.view.ViewConfiguration
import com.furnit.android.theme.PaafektColors
import com.furnit.android.utils.FurnitureClassNames
import com.furnit.android.utils.LogUtil
import java.text.NumberFormat
import kotlin.math.max
import kotlin.math.min

// Detection data for overlay display
data class DetectionResult(
    val x: Float,      // center x in input coords (640x640)
    val y: Float,      // center y in input coords
    val w: Float,      // width in input coords
    val h: Float,      // height in input coords
    val confidence: Float,
    val label: String,
    val classId: Int,
)

class FurnitureFitOverlayView(context: Context) : View(context) {
    private var maskBitmap: Bitmap? = null
    private var detections: List<DetectionResult> = emptyList()
    private var detectionClusters: List<List<Int>> = emptyList()
    private var inputSize = 640 // Model input size
    private var sourceFrameWidth = 640
    private var sourceFrameHeight = 640
    private var lastPrimaryLabel: String? = null
    private var hitTestPixels: IntArray? = null
    private var hitTestWidth = 0
    private var hitTestHeight = 0
    private var hitTestBitmap: Bitmap? = null
    private var showDetectionBoxes = false
    private var identifySelectionEnabled = false
    /** Pinned instances (matched each frame by class + IoU), not "all boxes of this class". */
    private var selectedPins: List<DetectionResult> = emptyList()
    private var pendingTappedDetection: DetectionResult? = null
    private var lastMaskDrawLogMs = 0L

    // Pinch-to-zoom scale factor for furniture (1.0 = neutral)
    private var furnitureScale = 1.0f
    /** AR-assisted overlay scale from the host (1f when AR is off or not yet valid). */
    private var assistedOverlayScale = 1f
    private var displayedFurnitureHeightMeters: Float? = null
    private var roomHeightMeters: Float? = null
    private var liveFrameAlignedOverlay = false
    private var translateX = 0f
    private var translateY = 0f

    // For single-finger drag
    private var initialTouchX = 0f
    private var initialTouchY = 0f
    private var lastTouchX = 0f
    private var lastTouchY = 0f
    private var isDraggingFurniture = false
    private var hasDraggedFurniture = false
    private var touchOnFurniture = false
    private val touchSlop = ViewConfiguration.get(context).scaledTouchSlop

    // Callback for when touch is outside furniture (for camera control)
    var onTouchOutsideFurniture: ((MotionEvent) -> Unit)? = null
    var onDetectionTapped: ((DetectionResult) -> Unit)? = null

    private val scaleGestureDetector = ScaleGestureDetector(context, ScaleListener())
    private val drawMatrix = Matrix()
    private val density = resources.displayMetrics.density
    private val bboxCornerRadiusPx = 6f * density

    private val maskSilhouettePaint = Paint().apply {
        isAntiAlias = true
        isFilterBitmap = true
    }

    // Swift DetectionBBoxOverlayView parity: white/yellow stroke, rounded corners (not green sharp rects).
    private val boxPaint = Paint().apply {
        color = Color.argb(224, 255, 255, 255)
        style = Paint.Style.STROKE
        strokeWidth = 1.2f * density
        isAntiAlias = true
    }

    private val selectedBoxPaint = Paint().apply {
        color = PaafektColors.accent
        style = Paint.Style.STROKE
        strokeWidth = 2.5f * density
        isAntiAlias = true
    }

    private val selectedTextBgPaint = Paint().apply {
        color = Color.argb(140, 0, 0, 0)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val textBgPaint = Paint().apply {
        color = Color.argb(97, 0, 0, 0)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 10f * density
        isAntiAlias = true
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.NORMAL)
    }

    private val selectedTextPaint = Paint().apply {
        color = Color.WHITE
        textSize = 11f * density
        isAntiAlias = true
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
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
        if (identifySelectionEnabled && maskBitmap == null) {
            return handleSelectionTouch(event)
        }
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                initialTouchX = event.x
                initialTouchY = event.y
                lastTouchX = event.x
                lastTouchY = event.y
                hasDraggedFurniture = false
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
                        hasDraggedFurniture = false
                        onTouchOutsideFurniture?.invoke(event)
                        return false
                    }
                    isDraggingFurniture = false
                    hasDraggedFurniture = true
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
                    if (!hasDraggedFurniture &&
                        (kotlin.math.abs(event.x - initialTouchX) >= touchSlop ||
                            kotlin.math.abs(event.y - initialTouchY) >= touchSlop)
                    ) {
                        hasDraggedFurniture = true
                    }
                    translateX += deltaX
                    translateY += deltaY
                    lastTouchX = event.x
                    lastTouchY = event.y
                    invalidate()
                }
            }
            MotionEvent.ACTION_UP -> {
                if (!touchOnFurniture) {
                    onTouchOutsideFurniture?.invoke(event)
                    return false
                }
                scaleGestureDetector.onTouchEvent(event)
                if (!hasDraggedFurniture) {
                    performClick()
                }
                val wasHandling = touchOnFurniture
                isDraggingFurniture = false
                hasDraggedFurniture = false
                touchOnFurniture = false
                return wasHandling
            }
            MotionEvent.ACTION_CANCEL -> {
                if (!touchOnFurniture) {
                    onTouchOutsideFurniture?.invoke(event)
                    return false
                }
                scaleGestureDetector.onTouchEvent(event)
                val wasHandling = touchOnFurniture
                isDraggingFurniture = false
                hasDraggedFurniture = false
                touchOnFurniture = false
                return wasHandling
            }
            else -> if (touchOnFurniture) scaleGestureDetector.onTouchEvent(event)
        }
        return touchOnFurniture
    }

    private fun handleSelectionTouch(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                pendingTappedDetection = findDetectionAt(event.x, event.y)
                if (pendingTappedDetection == null) {
                    onTouchOutsideFurniture?.invoke(event)
                    return false
                }
                return true
            }
            MotionEvent.ACTION_UP -> {
                val tappedDetection = pendingTappedDetection
                pendingTappedDetection = null
                if (tappedDetection != null) {
                    onDetectionTapped?.invoke(tappedDetection)
                    performClick()
                    return true
                }
                onTouchOutsideFurniture?.invoke(event)
                return false
            }
            MotionEvent.ACTION_CANCEL -> {
                pendingTappedDetection = null
                return false
            }
            MotionEvent.ACTION_MOVE -> {
                return pendingTappedDetection != null
            }
        }
        return pendingTappedDetection != null
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    /**
     * Check if the touch point is on the furniture: use detection bbox when available (furniture structure),
     * otherwise fall back to mask pixel (non-transparent).
     */
    private fun isTouchOnFurniture(touchX: Float, touchY: Float): Boolean {
        if (width == 0 || height == 0) return false
        if (findDetectionAt(touchX, touchY) != null) return true
        if (identifySelectionEnabled && maskBitmap == null) return false
        val detectionHit = findDetectionAt(touchX, touchY)
        if (detectionHit != null) return true
        if (detections.isNotEmpty() && inputSize > 0 && maskBitmap == null) return false
        return isTouchOnMask(touchX, touchY)
    }

    private data class DetectionTransform(
        val scaleX: Float,
        val scaleY: Float,
        val offsetX: Float,
        val offsetY: Float,
        val modelToSourceX: Float = 1f,
        val modelToSourceY: Float = 1f,
    )

    private data class Quad(
        val first: Float,
        val second: Float,
        val third: Float,
        val fourth: Float,
    )

    private fun currentDetectionTransform(): DetectionTransform {
        if (liveFrameAlignedOverlay) {
            val srcW = sourceFrameWidth.takeIf { it > 0 } ?: inputSize
            val srcH = sourceFrameHeight.takeIf { it > 0 } ?: inputSize
            val scale = max(width / srcW.toFloat(), height / srcH.toFloat())
            return DetectionTransform(
                scaleX = scale,
                scaleY = scale,
                offsetX = (width - srcW * scale) * 0.5f,
                offsetY = (height - srcH * scale) * 0.5f,
                modelToSourceX = srcW.toFloat() / inputSize.coerceAtLeast(1).toFloat(),
                modelToSourceY = srcH.toFloat() / inputSize.coerceAtLeast(1).toFloat(),
            )
        }

        val screenCenterX = width / 2f
        val screenCenterY = overlayScreenCenterY()
        val baseScale = min(width / inputSize.toFloat(), height / inputSize.toFloat())
        val totalScaleX = baseScale * furnitureScale * assistedOverlayScale
        val totalScaleY = totalScaleX * computeVerticalClampFactor(totalScaleX)
        return DetectionTransform(
            scaleX = totalScaleX,
            scaleY = totalScaleY,
            offsetX = screenCenterX - (inputSize / 2f) * totalScaleX + translateX,
            offsetY = screenCenterY - (inputSize / 2f) * totalScaleY + translateY,
        )
    }

    private fun rectForDetection(det: DetectionResult, transform: DetectionTransform): RectF {
        val leftModel = (det.x - det.w / 2f) * transform.modelToSourceX
        val topModel = (det.y - det.h / 2f) * transform.modelToSourceY
        val rightModel = (det.x + det.w / 2f) * transform.modelToSourceX
        val bottomModel = (det.y + det.h / 2f) * transform.modelToSourceY
        return RectF(
            leftModel * transform.scaleX + transform.offsetX,
            topModel * transform.scaleY + transform.offsetY,
            rightModel * transform.scaleX + transform.offsetX,
            bottomModel * transform.scaleY + transform.offsetY,
        )
    }

    private fun visibleClusterGroups(): List<List<Int>> {
        val normalized = detectionClusters
            .map { group -> group.filter { it in detections.indices }.distinct().sorted() }
            .filter { it.isNotEmpty() }
        return normalized.ifEmpty { detections.indices.map { listOf(it) } }
    }

    private fun representativeDetection(group: List<Int>): DetectionResult? {
        return group
            .mapNotNull { detections.getOrNull(it) }
            .maxWithOrNull(
                compareBy<DetectionResult> { it.confidence }
                    .thenBy { it.w * it.h }
            )
    }

    private fun clusterRect(group: List<Int>, transform: DetectionTransform): RectF? {
        var union: RectF? = null
        for (index in group) {
            val det = detections.getOrNull(index) ?: continue
            val rect = rectForDetection(det, transform)
            if (union == null) {
                union = RectF(rect)
            } else {
                union.union(rect)
            }
        }
        return union
    }

    private fun isClusterSelected(group: List<Int>): Boolean {
        return group.any { index ->
            val det = detections.getOrNull(index) ?: return@any false
            selectedPins.any { pin ->
                det.classId == pin.classId && iou(det, pin) >= 0.45f
            }
        }
    }

    private fun clusterLabel(group: List<Int>, representative: DetectionResult): String {
        val labels = group
            .mapNotNull { detections.getOrNull(it) }
            .map { FurnitureClassNames.localized(context, it.classId, it.label) }
            .distinct()
            .sorted()
        return if (labels.size > 1) {
            labels.joinToString(", ")
        } else {
            FurnitureClassNames.localized(context, representative.classId, representative.label)
        }
    }

    private fun findDetectionAt(touchX: Float, touchY: Float): DetectionResult? {
        if (width == 0 || height == 0) return null
        if (detections.isEmpty() || inputSize <= 0) return null
        val transform = currentDetectionTransform()
        return visibleClusterGroups()
            .mapNotNull { group ->
                val rect = clusterRect(group, transform) ?: return@mapNotNull null
                if (!rect.contains(touchX, touchY)) return@mapNotNull null
                val representative = representativeDetection(group) ?: return@mapNotNull null
                Triple(rect.width() * rect.height(), group, representative)
            }
            .minByOrNull { it.first }
            ?.third
    }

    private fun isTouchOnMask(touchX: Float, touchY: Float): Boolean {
        if (width == 0 || height == 0) return false
        val bmp = maskBitmap ?: return false
        val (totalScaleX, totalScaleY, maskLeft, maskTop) = if (liveFrameAlignedOverlay) {
            val scale = max(width / bmp.width.toFloat(), height / bmp.height.toFloat())
            Quad(scale, scale, (width - bmp.width * scale) * 0.5f, (height - bmp.height * scale) * 0.5f)
        } else {
            val screenCenterX = width / 2f
            val screenCenterY = overlayScreenCenterY()
            val baseScale = min(width / bmp.width.toFloat(), height / bmp.height.toFloat())
            val scaleX = baseScale * furnitureScale * assistedOverlayScale
            val scaleY = scaleX * computeVerticalClampFactor(scaleX)
            Quad(
                scaleX,
                scaleY,
                screenCenterX - (bmp.width / 2f) * scaleX + translateX,
                screenCenterY - (bmp.height / 2f) * scaleY + translateY,
            )
        }
        val bmpX = ((touchX - maskLeft) / totalScaleX).toInt()
        val bmpY = ((touchY - maskTop) / totalScaleY).toInt()
        if (bmpX < 0 || bmpX >= bmp.width || bmpY < 0 || bmpY >= bmp.height) return false
        updateHitTestCache(bmp)
        val pixels = hitTestPixels ?: return false
        val idx = bmpY * hitTestWidth + bmpX
        if (idx < 0 || idx >= pixels.size) return false
        return Color.alpha(pixels[idx]) > 10
    }

    private fun updateHitTestCache(bmp: Bitmap) {
        if (hitTestBitmap === bmp &&
            hitTestWidth == bmp.width &&
            hitTestHeight == bmp.height &&
            hitTestPixels != null
        ) {
            return
        }

        val pixels = IntArray(bmp.width * bmp.height)
        val readBitmap =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && bmp.config == Bitmap.Config.HARDWARE) {
                bmp.copy(Bitmap.Config.ARGB_8888, false)
            } else {
                bmp
            }

        try {
            readBitmap.getPixels(pixels, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        } finally {
            if (readBitmap !== bmp && !readBitmap.isRecycled) {
                readBitmap.recycle()
            }
        }

        hitTestPixels = pixels
        hitTestWidth = bmp.width
        hitTestHeight = bmp.height
        hitTestBitmap = bmp
    }

    private fun clearHitTestCache() {
        hitTestPixels = null
        hitTestWidth = 0
        hitTestHeight = 0
        hitTestBitmap = null
    }

    private fun replaceMaskBitmap(newMask: Bitmap?) {
        val oldMask = maskBitmap
        val alphaMask = newMask?.let { ensureAlphaMaskBitmap(it) }
        if (oldMask != null && oldMask !== newMask && oldMask !== alphaMask && !oldMask.isRecycled) {
            oldMask.recycle()
        }
        maskBitmap = alphaMask
        clearHitTestCache()
    }

    private fun ensureAlphaMaskBitmap(bitmap: Bitmap): Bitmap {
        if (bitmap.config == Bitmap.Config.ARGB_8888 && bitmap.hasAlpha()) {
            return bitmap
        }
        val copy = bitmap.copy(Bitmap.Config.ARGB_8888, false)
        copy.setHasAlpha(true)
        if (!bitmap.hasAlpha()) {
            LogUtil.w("FurnitureFitOverlay", "Incoming mask bitmap had hasAlpha=false; copied to ARGB_8888 alpha mask")
        }
        return copy
    }

    private fun maybeResetTransformForPrimaryDetection(dets: List<DetectionResult>) {
        val newPrimaryLabel = dets.firstOrNull()?.label
        if (newPrimaryLabel != null && newPrimaryLabel != lastPrimaryLabel) {
            furnitureScale = 1.0f
            translateX = 0f
            translateY = 0f
        }
        if (newPrimaryLabel != null) {
            lastPrimaryLabel = newPrimaryLabel
        }
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
        replaceMaskBitmap(b)
        invalidate()
    }

    fun setDetections(dets: List<DetectionResult>, modelInputSize: Int = 640) {
        maybeResetTransformForPrimaryDetection(dets)
        detections = dets
        detectionClusters = emptyList()
        inputSize = modelInputSize
        sourceFrameWidth = modelInputSize
        sourceFrameHeight = modelInputSize
        invalidate()
    }

    fun setDetectionBoxVisibility(visible: Boolean) {
        showDetectionBoxes = visible
        invalidate()
    }

    fun setIdentifySelectionState(enabled: Boolean, selectedPins: List<DetectionResult>) {
        identifySelectionEnabled = enabled
        this.selectedPins = selectedPins
        invalidate()
    }

    private fun iou(a: DetectionResult, b: DetectionResult): Float {
        val ax1 = a.x - a.w / 2f
        val ay1 = a.y - a.h / 2f
        val ax2 = a.x + a.w / 2f
        val ay2 = a.y + a.h / 2f
        val bx1 = b.x - b.w / 2f
        val by1 = b.y - b.h / 2f
        val bx2 = b.x + b.w / 2f
        val by2 = b.y + b.h / 2f
        val ix1 = max(ax1, bx1)
        val iy1 = max(ay1, by1)
        val ix2 = min(ax2, bx2)
        val iy2 = min(ay2, by2)
        val iw = max(0f, ix2 - ix1)
        val ih = max(0f, iy2 - iy1)
        val inter = iw * ih
        val ua = a.w * a.h + b.w * b.h - inter
        return if (ua > 0f) inter / ua else 0f
    }

    fun setMaskAndDetections(
        mask: Bitmap?,
        dets: List<DetectionResult>,
        modelInputSize: Int = 640,
        assistedScale: Float = 1f,
        displayedHeightMeters: Float? = null,
        roomHeightMeters: Float? = null,
        clusters: List<List<Int>> = emptyList(),
        frameAlignedOverlay: Boolean = false,
        sourceWidth: Int = mask?.width ?: modelInputSize,
        sourceHeight: Int = mask?.height ?: modelInputSize,
    ) {
        maybeResetTransformForPrimaryDetection(dets)
        replaceMaskBitmap(mask)
        detections = dets
        detectionClusters = clusters
        inputSize = modelInputSize
        sourceFrameWidth = sourceWidth.takeIf { it > 0 } ?: modelInputSize
        sourceFrameHeight = sourceHeight.takeIf { it > 0 } ?: modelInputSize
        liveFrameAlignedOverlay = frameAlignedOverlay
        this.displayedFurnitureHeightMeters = displayedHeightMeters
        this.roomHeightMeters = roomHeightMeters
        assistedOverlayScale = if (mask == null || frameAlignedOverlay) {
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
        // Uniform base scale to fit, scale around bitmap center, then translate to screen center.
        maskBitmap?.let { bmp ->
            val now = android.os.SystemClock.elapsedRealtime()
            if (now - lastMaskDrawLogMs >= 1000L) {
                lastMaskDrawLogMs = now
                LogUtil.i(
                    "FurnitureFitOverlay",
                    "Drawing mask bitmap: ${bmp.width}x${bmp.height} config=${bmp.config} hasAlpha=${bmp.hasAlpha()} " +
                        "overlayBg=${background != null} parentBg=${(parent as? View)?.background != null}",
                )
            }
            drawMatrix.reset()
            if (liveFrameAlignedOverlay) {
                val scale = max(width / bmp.width.toFloat(), height / bmp.height.toFloat())
                drawMatrix.postScale(scale, scale)
                drawMatrix.postTranslate(
                    (width - bmp.width * scale) * 0.5f,
                    (height - bmp.height * scale) * 0.5f,
                )
            } else {
                val baseScale = min(width / bmp.width.toFloat(), height / bmp.height.toFloat())
                val totalScaleX = baseScale * furnitureScale * assistedOverlayScale
                val totalScaleY = totalScaleX * computeVerticalClampFactor(totalScaleX)
                drawMatrix.postScale(totalScaleX, totalScaleY, bmp.width / 2f, bmp.height / 2f)
                val screenCenterX = width / 2f
                val screenCenterY = overlayScreenCenterY()
                drawMatrix.postTranslate(screenCenterX - bmp.width / 2f, screenCenterY - bmp.height / 2f)
                drawMatrix.postTranslate(translateX, translateY)
            }

            canvas.drawBitmap(bmp, drawMatrix, maskSilhouettePaint)
        }

        val shouldDrawDetectionBoxes = showDetectionBoxes && detections.isNotEmpty()
        if (shouldDrawDetectionBoxes) {
            val transform = currentDetectionTransform()

            for (group in visibleClusterGroups()) {
                val rect = clusterRect(group, transform) ?: continue
                val representative = representativeDetection(group) ?: continue
                val isSelected = isClusterSelected(group)
                val activeBoxPaint = if (isSelected) selectedBoxPaint else boxPaint

                canvas.drawRoundRect(rect, bboxCornerRadiusPx, bboxCornerRadiusPx, activeBoxPaint)

                if (!BuildConfig.DEBUG) continue

                val activeTextBgPaint = if (isSelected) selectedTextBgPaint else textBgPaint
                val activeTextPaint = if (isSelected) selectedTextPaint else textPaint
                val scoreText = NumberFormat.getNumberInstance(
                    resources.configuration.locales[0],
                ).apply {
                    minimumFractionDigits = 2
                    maximumFractionDigits = 2
                }.format(representative.confidence.toDouble())
                val label = clusterLabel(group, representative)
                val text = if (label.isEmpty()) "" else "$label $scoreText"
                if (text.isEmpty()) continue

                val textWidth = activeTextPaint.measureText(text)
                val textHeight = activeTextPaint.textSize
                val maxLabelWidth = min(max(rect.width(), 56f * density), 140f * density)
                val bgLeft = rect.left
                val bgTop = max(0f, rect.top - textHeight - 8f * density)
                val bgRight = rect.left + min(maxLabelWidth, textWidth + 10f * density)
                val bgBottom = rect.top
                canvas.drawRoundRect(
                    bgLeft,
                    bgTop,
                    bgRight,
                    bgBottom,
                    bboxCornerRadiusPx,
                    bboxCornerRadiusPx,
                    activeTextBgPaint,
                )
                canvas.drawText(
                    text,
                    rect.left + 5f * density,
                    rect.top - 5f * density,
                    activeTextPaint,
                )
            }
        }
    }
}
