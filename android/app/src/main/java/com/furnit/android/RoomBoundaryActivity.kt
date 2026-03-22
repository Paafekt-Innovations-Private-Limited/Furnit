package com.furnit.android

import android.content.Context
import android.content.Intent
import android.content.pm.ActivityInfo
import android.graphics.*
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import com.furnit.android.utils.CrashReporter
import com.furnit.android.utils.LogUtil
import android.view.*
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import com.furnit.android.models.PhotoOrientation
import com.furnit.android.models.RoomStructure
import com.furnit.android.services.SinglePhotoRoomReconstructor
import java.io.File
import java.io.InputStream
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * RoomBoundaryActivity - Boundary adjustment view for manual room creation
 * (Matches Swift's RoomBoundaryDetectionView)
 *
 * Shows the selected image with draggable boundary lines:
 * - Floor line (green) - horizontal
 * - Ceiling line (cyan) - horizontal
 * - Left wall line (red) - vertical
 * - Right wall line (yellow) - vertical
 * - Vanishing point (magenta) - draggable point
 */
class RoomBoundaryActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_IMAGE_URI = "image_uri"
        const val EXTRA_PHOTO_ORIENTATION = "photo_orientation"
        const val RESULT_BOUNDARIES = "boundaries"
        const val REQUEST_CODE = 1001
    }

    private var imageBitmap: Bitmap? = null
    private lateinit var boundaryView: BoundaryOverlayView
    private lateinit var imageView: ImageView
    private lateinit var rootLayout: FrameLayout
    private var progressOverlay: View? = null

    // Boundary values (percentages 0-1)
    private var structure = RoomStructure()

    // Photo orientation (landscape vs portrait)
    private var photoOrientation: PhotoOrientation = PhotoOrientation.PORTRAIT
    private val isLandscape: Boolean get() = photoOrientation == PhotoOrientation.LANDSCAPE

    // Room reconstructor
    private lateinit var reconstructor: SinglePhotoRoomReconstructor
    private val mainHandler = Handler(Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val imageUriString = intent.getStringExtra(EXTRA_IMAGE_URI)
        if (imageUriString == null) {
            LogUtil.e("RoomBoundary", "No image URI provided")
            finish()
            return
        }

        // Get photo orientation and lock screen accordingly
        val orientationStr = intent.getStringExtra(EXTRA_PHOTO_ORIENTATION) ?: "portrait"
        photoOrientation = when (orientationStr) {
            "landscape" -> PhotoOrientation.LANDSCAPE
            "square" -> PhotoOrientation.SQUARE
            else -> PhotoOrientation.PORTRAIT
        }
        LogUtil.d("RoomBoundary", "Photo orientation: ${photoOrientation.value}")

        // Lock screen orientation to match photo orientation (like iOS)
        requestedOrientation = if (isLandscape) {
            ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE
        } else {
            ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        }

        val imageUri = Uri.parse(imageUriString)
        loadImage(imageUri)

        if (imageBitmap == null) {
            Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        reconstructor = SinglePhotoRoomReconstructor(this)
        setupUI()
    }

    override fun onDestroy() {
        super.onDestroy()
        // Unlock orientation when leaving
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
    }

    private fun loadImage(uri: Uri) {
        try {
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            imageBitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()
            LogUtil.d("RoomBoundary", "Image loaded: ${imageBitmap?.width}x${imageBitmap?.height}")
        } catch (e: Exception) {
            LogUtil.e("RoomBoundary", "Failed to load image", e)
            CrashReporter.report(this, e, "Room boundary — load image")
        }
    }

    private fun setupUI() {
        rootLayout = FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
        }

        if (isLandscape) {
            setupLandscapeUI()
        } else {
            setupPortraitUI()
        }

        setContentView(rootLayout)

        // Post to get actual image bounds after layout
        imageView.post {
            boundaryView.setImageBounds(getImageBoundsInView())
        }
    }

    /**
     * Portrait layout: vertical with top bar, image area, and bottom controls
     */
    private fun setupPortraitUI() {
        val mainLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }

        // Top bar with back button and title
        val topBar = createTopBar()
        mainLayout.addView(topBar)

        // Image container with boundary overlay
        val imageContainer = createImageContainer()
        mainLayout.addView(imageContainer, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            0, 1f
        ))

        // Instructions panel
        val instructionsPanel = createControlsPanel(isHorizontal = false)
        mainLayout.addView(instructionsPanel)

        rootLayout.addView(mainLayout)
    }

    /**
     * Landscape layout: full-screen image with horizontal bottom overlay bar
     * Matches iOS RoomBoundaryDetectionView landscape layout
     */
    private fun setupLandscapeUI() {
        // Full-screen image container
        val imageContainer = createImageContainer()
        rootLayout.addView(imageContainer, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))

        // Horizontal bottom overlay bar with controls
        val bottomBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(Color.parseColor("#CC2A2A2A"))
            gravity = Gravity.CENTER_VERTICAL
            setPadding(24, 12, 24, 12)
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM
            )
        }

        // Back button
        val backBtn = TextView(this).apply {
            text = getString(R.string.boundary_back)
            textSize = 14f
            setTextColor(Color.parseColor("#007AFF"))
            setPadding(0, 0, 24, 0)
            setOnClickListener { finish() }
        }
        bottomBar.addView(backBtn)

        // Legend (compact, horizontal)
        val legendLayout = createLegendLayout()
        bottomBar.addView(legendLayout, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))

        // Reset button
        val resetBtn = Button(this).apply {
            text = getString(R.string.common_reset)
            textSize = 12f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#555555"))
            setPadding(20, 8, 20, 8)
            minimumHeight = 0
            minHeight = 0
            setOnClickListener {
                structure.reset()
                boundaryView.updateStructure(structure)
            }
        }
        bottomBar.addView(resetBtn, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { setMargins(8, 0, 8, 0) })

        // Done button
        val doneBtn = Button(this).apply {
            text = getString(R.string.common_done)
            textSize = 12f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#4CAF50"))
            setPadding(28, 8, 28, 8)
            minimumHeight = 0
            minHeight = 0
            setOnClickListener {
                onDonePressed()
            }
        }
        bottomBar.addView(doneBtn)

        rootLayout.addView(bottomBar)
    }

    private fun createTopBar(): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(Color.parseColor("#1A1A1A"))
            setPadding(16, 48, 16, 16)
            gravity = Gravity.CENTER_VERTICAL

            val backBtn = TextView(this@RoomBoundaryActivity).apply {
                text = getString(R.string.boundary_back)
                textSize = 16f
                setTextColor(Color.parseColor("#007AFF"))
                setOnClickListener { finish() }
            }
            addView(backBtn)

            val title = TextView(this@RoomBoundaryActivity).apply {
                text = getString(R.string.boundary_adjust)
                textSize = 18f
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            }
            addView(title)

            // Orientation indicator
            val orientationLabel = TextView(this@RoomBoundaryActivity).apply {
                text = if (isLandscape) "\uD83D\uDCF1↔ Landscape" else "\uD83D\uDCF1↕ Portrait"
                textSize = 12f
                setTextColor(Color.GRAY)
            }
            addView(orientationLabel)
        }
    }

    private fun createImageContainer(): FrameLayout {
        val container = FrameLayout(this)

        imageView = ImageView(this).apply {
            setImageBitmap(imageBitmap)
            scaleType = ImageView.ScaleType.FIT_CENTER
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }
        container.addView(imageView)

        boundaryView = BoundaryOverlayView(this, structure).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            onBoundaryChanged = { newStructure ->
                structure = newStructure
            }
        }
        container.addView(boundaryView)

        return container
    }

    private fun createLegendLayout(): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL

            fun addLegendDot(color: Int, label: String) {
                val dot = View(this@RoomBoundaryActivity).apply {
                    layoutParams = LinearLayout.LayoutParams(12, 12).apply {
                        setMargins(6, 0, 2, 0)
                    }
                    setBackgroundColor(color)
                }
                addView(dot)
                val text = TextView(this@RoomBoundaryActivity).apply {
                    this.text = label
                    textSize = 10f
                    setTextColor(Color.LTGRAY)
                }
                addView(text)
            }

            addLegendDot(Color.GREEN, "F")
            addLegendDot(Color.CYAN, "C")
            addLegendDot(Color.RED, "L")
            addLegendDot(Color.YELLOW, "R")
            addLegendDot(Color.MAGENTA, "VP")
        }
    }

    private fun createControlsPanel(isHorizontal: Boolean): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#2A2A2A"))
            setPadding(16, 8, 16, 8)

            // Combined legend and buttons in one row
            val controlsRow = LinearLayout(this@RoomBoundaryActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL

                // Legend (compact)
                val legendLayout = createLegendLayout()
                legendLayout.layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                addView(legendLayout)

                // Buttons
                val resetBtn = Button(this@RoomBoundaryActivity).apply {
                    text = getString(R.string.common_reset)
                    textSize = 12f
                    setTextColor(Color.WHITE)
                    setBackgroundColor(Color.parseColor("#555555"))
                    setPadding(24, 8, 24, 8)
                    minimumHeight = 0
                    minHeight = 0
                    setOnClickListener {
                        structure.reset()
                        boundaryView.updateStructure(structure)
                    }
                }
                addView(resetBtn, LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                ).apply { setMargins(8, 0, 8, 0) })

                val doneBtn = Button(this@RoomBoundaryActivity).apply {
                    text = getString(R.string.common_done)
                    textSize = 12f
                    setTextColor(Color.WHITE)
                    setBackgroundColor(Color.parseColor("#4CAF50"))
                    setPadding(32, 8, 32, 8)
                    minimumHeight = 0
                    minHeight = 0
                    setOnClickListener {
                        onDonePressed()
                    }
                }
                addView(doneBtn, LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                ))
            }
            addView(controlsRow)

            // Instruction hint
            val instructionText = TextView(this@RoomBoundaryActivity).apply {
                text = getString(R.string.boundary_drag_handles)
                textSize = 11f
                setTextColor(Color.GRAY)
                gravity = Gravity.CENTER
                setPadding(0, 4, 0, 0)
            }
            addView(instructionText)
        }
    }

    private fun getImageBoundsInView(): RectF {
        val bitmap = imageBitmap ?: return RectF()
        val imageView = this.imageView

        val viewWidth = imageView.width.toFloat()
        val viewHeight = imageView.height.toFloat()
        val imageWidth = bitmap.width.toFloat()
        val imageHeight = bitmap.height.toFloat()

        val imageAspect = imageWidth / imageHeight
        val viewAspect = viewWidth / viewHeight

        val displayWidth: Float
        val displayHeight: Float
        val offsetX: Float
        val offsetY: Float

        if (imageAspect > viewAspect) {
            displayWidth = viewWidth
            displayHeight = viewWidth / imageAspect
            offsetX = 0f
            offsetY = (viewHeight - displayHeight) / 2
        } else {
            displayHeight = viewHeight
            displayWidth = viewHeight * imageAspect
            offsetX = (viewWidth - displayWidth) / 2
            offsetY = 0f
        }

        return RectF(offsetX, offsetY, offsetX + displayWidth, offsetY + displayHeight)
    }

    private fun onDonePressed() {
        LogUtil.d("RoomBoundary", "Done pressed with boundaries:")
        LogUtil.d("RoomBoundary", "  Floor: ${structure.floorY}, Ceiling: ${structure.ceilingY}")
        LogUtil.d("RoomBoundary", "  Left: ${structure.leftX}, Right: ${structure.rightX}")
        LogUtil.d("RoomBoundary", "  VP: (${structure.vanishingX}, ${structure.vanishingY})")

        val bitmap = imageBitmap
        if (bitmap == null) {
            Toast.makeText(this, getString(R.string.boundary_no_image), Toast.LENGTH_SHORT).show()
            return
        }

        // Show progress overlay
        showProgressOverlay()

        // Process with reconstructor
        reconstructor.processPhotoWithBoundaries(
            bitmap,
            structure,
            SinglePhotoRoomReconstructor.RoomDimensions(),
            object : SinglePhotoRoomReconstructor.ProgressCallback {
                override fun onProgress(progress: Float, message: String) {
                    mainHandler.post {
                        updateProgress(progress, message)
                    }
                }

                override fun onComplete(glbFile: File?) {
                    mainHandler.post {
                        hideProgressOverlay()
                        if (glbFile != null) {
                            // Check if GLB file was created for 3D preview
                            if (glbFile.name.endsWith(".glb")) {
                                // Navigate to WebGL-based GLBRoomActivity for preview (matching iOS)
                                val intent = Intent(this@RoomBoundaryActivity, GLBRoomActivity::class.java)
                                intent.putExtra(GLBRoomActivity.EXTRA_GLB_PATH, glbFile.absolutePath)
                                intent.putExtra(GLBRoomActivity.EXTRA_ROOM_NAME, "Your Room")
                                intent.putExtra(GLBRoomActivity.EXTRA_IS_PREVIEW, true)
                                intent.putExtra(GLBRoomActivity.EXTRA_PHOTO_ORIENTATION, photoOrientation.value)
                                startActivity(intent)
                            } else {
                                // Fallback to 2D room viewer
                                val intent = Intent(this@RoomBoundaryActivity, RoomViewerActivity::class.java)
                                intent.putExtra(RoomViewerActivity.EXTRA_ROOM_FOLDER, glbFile.parentFile?.absolutePath)
                                startActivity(intent)
                            }
                            finish()
                        } else {
                            Toast.makeText(this@RoomBoundaryActivity, getString(R.string.boundary_failed_create), Toast.LENGTH_SHORT).show()
                        }
                    }
                }

                override fun onError(message: String) {
                    mainHandler.post {
                        hideProgressOverlay()
                        Toast.makeText(this@RoomBoundaryActivity, message, Toast.LENGTH_LONG).show()
                    }
                }
            }
        )
    }

    private var progressText: TextView? = null
    private var progressBar: ProgressBar? = null

    private fun showProgressOverlay() {
        progressOverlay = FrameLayout(this).apply {
            setBackgroundColor(Color.parseColor("#CC000000"))
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )

            val container = LinearLayout(this@RoomBoundaryActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setBackgroundColor(Color.parseColor("#333333"))
                setPadding(48, 48, 48, 48)
                layoutParams = FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    Gravity.CENTER
                )
            }

            progressBar = ProgressBar(this@RoomBoundaryActivity, null, android.R.attr.progressBarStyleHorizontal).apply {
                layoutParams = LinearLayout.LayoutParams(300, ViewGroup.LayoutParams.WRAP_CONTENT)
                max = 100
                progress = 0
            }
            container.addView(progressBar)

            progressText = TextView(this@RoomBoundaryActivity).apply {
                text = getString(R.string.boundary_processing)
                textSize = 16f
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
                setPadding(0, 24, 0, 0)
            }
            container.addView(progressText)

            addView(container)
        }
        rootLayout.addView(progressOverlay)
    }

    private fun updateProgress(progress: Float, message: String) {
        progressBar?.progress = (progress * 100).toInt()
        progressText?.text = message
    }

    private fun hideProgressOverlay() {
        progressOverlay?.let {
            rootLayout.removeView(it)
        }
        progressOverlay = null
    }
}

/**
 * Custom view for drawing and interacting with boundary lines
 */
class BoundaryOverlayView(
    context: Context,
    private var structure: RoomStructure
) : View(context) {

    var onBoundaryChanged: ((RoomStructure) -> Unit)? = null

    private var imageBounds = RectF()
    private val linePaint = Paint().apply {
        strokeWidth = 6f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    private val handlePaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    private val handleRadius = 30f
    private val touchRadius = 60f

    private enum class DragHandle {
        NONE, FLOOR, CEILING, LEFT, RIGHT, VANISHING_POINT
    }
    private var activeDrag = DragHandle.NONE

    fun setImageBounds(bounds: RectF) {
        imageBounds = bounds
        invalidate()
    }

    fun updateStructure(newStructure: RoomStructure) {
        structure = newStructure
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (imageBounds.isEmpty) return

        val left = imageBounds.left
        val top = imageBounds.top
        val width = imageBounds.width()
        val height = imageBounds.height()

        // Floor line (GREEN)
        val floorY = top + structure.floorY * height
        linePaint.color = Color.GREEN
        canvas.drawLine(left, floorY, left + width, floorY, linePaint)
        handlePaint.color = Color.GREEN
        canvas.drawCircle(left + width / 2, floorY, handleRadius, handlePaint)

        // Ceiling line (CYAN)
        val ceilingY = top + structure.ceilingY * height
        linePaint.color = Color.CYAN
        canvas.drawLine(left, ceilingY, left + width, ceilingY, linePaint)
        handlePaint.color = Color.CYAN
        canvas.drawCircle(left + width / 2, ceilingY, handleRadius, handlePaint)

        // Left wall line (RED)
        val leftX = left + structure.leftX * width
        linePaint.color = Color.RED
        canvas.drawLine(leftX, top, leftX, top + height, linePaint)
        handlePaint.color = Color.RED
        canvas.drawCircle(leftX, top + height / 2, handleRadius, handlePaint)

        // Right wall line (YELLOW)
        val rightX = left + structure.rightX * width
        linePaint.color = Color.YELLOW
        canvas.drawLine(rightX, top, rightX, top + height, linePaint)
        handlePaint.color = Color.YELLOW
        canvas.drawCircle(rightX, top + height / 2, handleRadius, handlePaint)

        // Vanishing point (MAGENTA)
        val vpX = left + structure.vanishingX * width
        val vpY = top + structure.vanishingY * height
        handlePaint.color = Color.MAGENTA
        canvas.drawCircle(vpX, vpY, handleRadius * 1.2f, handlePaint)
        // Crosshair
        linePaint.color = Color.WHITE
        linePaint.strokeWidth = 3f
        canvas.drawLine(vpX - 25, vpY, vpX + 25, vpY, linePaint)
        canvas.drawLine(vpX, vpY - 25, vpX, vpY + 25, linePaint)
        linePaint.strokeWidth = 6f
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (imageBounds.isEmpty) return false

        val x = event.x
        val y = event.y

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                activeDrag = findNearestHandle(x, y)
                return activeDrag != DragHandle.NONE
            }
            MotionEvent.ACTION_MOVE -> {
                if (activeDrag != DragHandle.NONE) {
                    updateBoundary(x, y)
                    invalidate()
                    return true
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                activeDrag = DragHandle.NONE
            }
        }
        return super.onTouchEvent(event)
    }

    private fun findNearestHandle(x: Float, y: Float): DragHandle {
        val left = imageBounds.left
        val top = imageBounds.top
        val width = imageBounds.width()
        val height = imageBounds.height()

        // Check vanishing point first (highest priority)
        val vpX = left + structure.vanishingX * width
        val vpY = top + structure.vanishingY * height
        if (distance(x, y, vpX, vpY) < touchRadius) return DragHandle.VANISHING_POINT

        // Check floor handle
        val floorY = top + structure.floorY * height
        val floorHandleX = left + width / 2
        if (distance(x, y, floorHandleX, floorY) < touchRadius) return DragHandle.FLOOR

        // Check ceiling handle
        val ceilingY = top + structure.ceilingY * height
        val ceilingHandleX = left + width / 2
        if (distance(x, y, ceilingHandleX, ceilingY) < touchRadius) return DragHandle.CEILING

        // Check left handle
        val leftX = left + structure.leftX * width
        val leftHandleY = top + height / 2
        if (distance(x, y, leftX, leftHandleY) < touchRadius) return DragHandle.LEFT

        // Check right handle
        val rightX = left + structure.rightX * width
        val rightHandleY = top + height / 2
        if (distance(x, y, rightX, rightHandleY) < touchRadius) return DragHandle.RIGHT

        return DragHandle.NONE
    }

    private fun updateBoundary(x: Float, y: Float) {
        val left = imageBounds.left
        val top = imageBounds.top
        val width = imageBounds.width()
        val height = imageBounds.height()

        when (activeDrag) {
            DragHandle.FLOOR -> {
                val newY = (y - top) / height
                structure.floorY = min(max(newY, 0.5f), 0.95f)
            }
            DragHandle.CEILING -> {
                val newY = (y - top) / height
                structure.ceilingY = min(max(newY, 0.05f), 0.5f)
            }
            DragHandle.LEFT -> {
                val newX = (x - left) / width
                structure.leftX = min(max(newX, 0.02f), 0.4f)
            }
            DragHandle.RIGHT -> {
                val newX = (x - left) / width
                structure.rightX = min(max(newX, 0.6f), 0.98f)
            }
            DragHandle.VANISHING_POINT -> {
                val newX = (x - left) / width
                val newY = (y - top) / height
                structure.vanishingX = min(max(newX, 0.1f), 0.9f)
                structure.vanishingY = min(max(newY, 0.1f), 0.9f)
            }
            DragHandle.NONE -> {}
        }

        onBoundaryChanged?.invoke(structure)
    }

    private fun distance(x1: Float, y1: Float, x2: Float, y2: Float): Float {
        val dx = x1 - x2
        val dy = y1 - y2
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }
}
