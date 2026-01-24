package com.furnit.android

import android.content.Context
import android.content.Intent
import android.graphics.*
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.*
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
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

    // Room reconstructor
    private lateinit var reconstructor: SinglePhotoRoomReconstructor
    private val mainHandler = Handler(Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val imageUriString = intent.getStringExtra(EXTRA_IMAGE_URI)
        if (imageUriString == null) {
            Log.e("RoomBoundary", "No image URI provided")
            finish()
            return
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

    private fun loadImage(uri: Uri) {
        try {
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            imageBitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()
            Log.d("RoomBoundary", "Image loaded: ${imageBitmap?.width}x${imageBitmap?.height}")
        } catch (e: Exception) {
            Log.e("RoomBoundary", "Failed to load image", e)
        }
    }

    private fun setupUI() {
        rootLayout = FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
        }

        // Main vertical layout
        val mainLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }

        // Top bar with back button and title
        val topBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(Color.parseColor("#1A1A1A"))
            setPadding(16, 48, 16, 16)
            gravity = Gravity.CENTER_VERTICAL

            val backBtn = TextView(this@RoomBoundaryActivity).apply {
                text = "< Back"
                textSize = 16f
                setTextColor(Color.parseColor("#007AFF"))
                setOnClickListener { finish() }
            }
            addView(backBtn)

            val title = TextView(this@RoomBoundaryActivity).apply {
                text = "Adjust Boundaries"
                textSize = 18f
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            }
            addView(title)

            // Spacer for symmetry
            val spacer = View(this@RoomBoundaryActivity).apply {
                layoutParams = LinearLayout.LayoutParams(80, 1)
            }
            addView(spacer)
        }
        mainLayout.addView(topBar)

        // Image container with boundary overlay
        val imageContainer = FrameLayout(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0, 1f
            )
        }

        imageView = ImageView(this).apply {
            setImageBitmap(imageBitmap)
            scaleType = ImageView.ScaleType.FIT_CENTER
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }
        imageContainer.addView(imageView)

        boundaryView = BoundaryOverlayView(this, structure).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            onBoundaryChanged = { newStructure ->
                structure = newStructure
            }
        }
        imageContainer.addView(boundaryView)

        mainLayout.addView(imageContainer)

        // Instructions panel
        val instructionsPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#2A2A2A"))
            setPadding(24, 16, 24, 16)

            val instructionText = TextView(this@RoomBoundaryActivity).apply {
                text = "Drag the handles to adjust room boundaries"
                textSize = 16f
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
            }
            addView(instructionText)

            // Legend
            val legendLayout = LinearLayout(this@RoomBoundaryActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER
                setPadding(0, 12, 0, 12)

                fun addLegendItem(color: Int, label: String) {
                    val item = LinearLayout(this@RoomBoundaryActivity).apply {
                        orientation = LinearLayout.HORIZONTAL
                        gravity = Gravity.CENTER_VERTICAL
                        setPadding(12, 0, 12, 0)

                        val dot = View(this@RoomBoundaryActivity).apply {
                            layoutParams = LinearLayout.LayoutParams(16, 16).apply {
                                setMargins(0, 0, 8, 0)
                            }
                            setBackgroundColor(color)
                        }
                        addView(dot)

                        val text = TextView(this@RoomBoundaryActivity).apply {
                            this.text = label
                            textSize = 12f
                            setTextColor(Color.LTGRAY)
                        }
                        addView(text)
                    }
                    addView(item)
                }

                addLegendItem(Color.GREEN, "Floor")
                addLegendItem(Color.CYAN, "Ceiling")
                addLegendItem(Color.RED, "Left")
                addLegendItem(Color.YELLOW, "Right")
                addLegendItem(Color.MAGENTA, "VP")
            }
            addView(legendLayout)

            // Buttons
            val buttonLayout = LinearLayout(this@RoomBoundaryActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER
                setPadding(0, 8, 0, 8)

                val resetBtn = Button(this@RoomBoundaryActivity).apply {
                    text = "Reset"
                    setTextColor(Color.WHITE)
                    setBackgroundColor(Color.parseColor("#555555"))
                    setPadding(32, 16, 32, 16)
                    setOnClickListener {
                        structure.reset()
                        boundaryView.updateStructure(structure)
                    }
                }
                addView(resetBtn, LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                ).apply { setMargins(16, 0, 16, 0) })

                val doneBtn = Button(this@RoomBoundaryActivity).apply {
                    text = "Done"
                    setTextColor(Color.WHITE)
                    setBackgroundColor(Color.parseColor("#4CAF50"))
                    setPadding(48, 16, 48, 16)
                    setOnClickListener {
                        onDonePressed()
                    }
                }
                addView(doneBtn, LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                ).apply { setMargins(16, 0, 16, 0) })
            }
            addView(buttonLayout)
        }
        mainLayout.addView(instructionsPanel)

        rootLayout.addView(mainLayout)
        setContentView(rootLayout)

        // Post to get actual image bounds after layout
        imageView.post {
            boundaryView.setImageBounds(getImageBoundsInView())
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
        Log.d("RoomBoundary", "Done pressed with boundaries:")
        Log.d("RoomBoundary", "  Floor: ${structure.floorY}, Ceiling: ${structure.ceilingY}")
        Log.d("RoomBoundary", "  Left: ${structure.leftX}, Right: ${structure.rightX}")
        Log.d("RoomBoundary", "  VP: (${structure.vanishingX}, ${structure.vanishingY})")

        val bitmap = imageBitmap
        if (bitmap == null) {
            Toast.makeText(this, "No image available", Toast.LENGTH_SHORT).show()
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
                                // Navigate to 3D ModelDetailActivity for preview
                                val intent = Intent(this@RoomBoundaryActivity, ModelDetailActivity::class.java)
                                intent.putExtra(ModelDetailActivity.EXTRA_GLB_PATH, glbFile.absolutePath)
                                intent.putExtra(ModelDetailActivity.EXTRA_ROOM_NAME, "Your Room")
                                intent.putExtra(ModelDetailActivity.EXTRA_IS_PREVIEW, true)
                                startActivity(intent)
                            } else {
                                // Fallback to 2D room viewer
                                val intent = Intent(this@RoomBoundaryActivity, RoomViewerActivity::class.java)
                                intent.putExtra(RoomViewerActivity.EXTRA_ROOM_FOLDER, glbFile.parentFile?.absolutePath)
                                startActivity(intent)
                            }
                            finish()
                        } else {
                            Toast.makeText(this@RoomBoundaryActivity, "Failed to create room", Toast.LENGTH_SHORT).show()
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
                text = "Processing..."
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
