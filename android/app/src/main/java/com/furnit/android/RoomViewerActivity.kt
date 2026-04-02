package com.furnit.android

import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Typeface
import android.os.Bundle
import com.furnit.android.utils.LogUtil
import android.view.Gravity
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsControllerCompat
import java.io.File
import kotlin.math.max
import kotlin.math.min

/**
 * RoomViewerActivity - Displays the created 3D room
 * (Matches Swift's SceneKitViewer - simplified preview version)
 *
 * Shows the extracted room textures as a preview
 */
class RoomViewerActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_ROOM_FOLDER = "room_folder"
    }

    // For pinch-to-zoom and pan
    private lateinit var scaleGestureDetector: ScaleGestureDetector
    private var scaleFactor = 1.0f
    private var translateX = 0f
    private var translateY = 0f
    private var lastTouchX = 0f
    private var lastTouchY = 0f
    private var activePointerId = MotionEvent.INVALID_POINTER_ID
    private lateinit var mainImageView: ImageView
    private val imageMatrix = Matrix()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Enable true edge-to-edge display (matching iOS ignoresSafeArea)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        window.statusBarColor = Color.TRANSPARENT
        window.navigationBarColor = Color.TRANSPARENT

        WindowInsetsControllerCompat(window, window.decorView).let { controller ->
            controller.isAppearanceLightStatusBars = false
            controller.isAppearanceLightNavigationBars = false
        }

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
            window.attributes.layoutInDisplayCutoutMode =
                WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
        }

        LogUtil.d("RoomViewer", "onCreate called")

        val roomFolderPath = intent.getStringExtra(EXTRA_ROOM_FOLDER)
        LogUtil.d("RoomViewer", "Room folder path from intent: $roomFolderPath")

        if (roomFolderPath == null) {
            LogUtil.e("RoomViewer", "No room folder provided")
            Toast.makeText(this, getString(R.string.room_viewer_no_folder), Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        val roomFolder = File(roomFolderPath)
        LogUtil.d("RoomViewer", "Room folder exists: ${roomFolder.exists()}, isDirectory: ${roomFolder.isDirectory}")

        if (!roomFolder.exists()) {
            LogUtil.e("RoomViewer", "Room folder doesn't exist: $roomFolderPath")
            Toast.makeText(this, getString(R.string.room_viewer_folder_not_found), Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        // List files in folder for debugging
        roomFolder.listFiles()?.forEach { file ->
            LogUtil.d("RoomViewer", "  File: ${file.name}")
        }

        // Initialize scale gesture detector
        scaleGestureDetector = ScaleGestureDetector(this, ScaleListener())

        setupUI(roomFolder)
    }

    private inner class ScaleListener : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector: ScaleGestureDetector): Boolean {
            scaleFactor *= detector.scaleFactor
            scaleFactor = max(0.5f, min(scaleFactor, 5.0f)) // Limit zoom between 0.5x and 5x
            updateImageTransform()
            return true
        }
    }

    private fun updateImageTransform() {
        imageMatrix.reset()
        imageMatrix.postScale(scaleFactor, scaleFactor, mainImageView.width / 2f, mainImageView.height / 2f)
        imageMatrix.postTranslate(translateX, translateY)
        mainImageView.imageMatrix = imageMatrix
    }

    private fun setupUI(roomFolder: File) {
        val rootLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#1A1A1A"))
        }

        // Top bar
        val topBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(Color.parseColor("#2A2A2A"))
            setPadding(16, 48, 16, 16)
            gravity = Gravity.CENTER_VERTICAL

            val backBtn = TextView(this@RoomViewerActivity).apply {
                text = getString(R.string.photo_room_back)
                textSize = 16f
                setTextColor(Color.parseColor("#007AFF"))
                setOnClickListener { finish() }
            }
            addView(backBtn)

            val title = TextView(this@RoomViewerActivity).apply {
                text = getString(R.string.room_viewer_your_room)
                textSize = 18f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            }
            addView(title)

            val saveBtn = TextView(this@RoomViewerActivity).apply {
                text = getString(R.string.common_save)
                textSize = 16f
                setTextColor(Color.parseColor("#4CAF50"))
                setOnClickListener {
                    Toast.makeText(this@RoomViewerActivity, getString(R.string.room_viewer_saved_toast), Toast.LENGTH_SHORT).show()
                    finish()
                }
            }
            addView(saveBtn)
        }
        rootLayout.addView(topBar)

        // Main content - front wall preview (the main room view) with pinch-to-zoom
        val frontWallFile = File(roomFolder, "front_wall.png")
        if (frontWallFile.exists()) {
            val bitmap = BitmapFactory.decodeFile(frontWallFile.absolutePath)
            mainImageView = ImageView(this).apply {
                setImageBitmap(bitmap)
                scaleType = ImageView.ScaleType.MATRIX
                setBackgroundColor(Color.BLACK)
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    0, 1f
                )
            }

            // Add touch listener for pinch-to-zoom and pan
            mainImageView.setOnTouchListener { _, event ->
                scaleGestureDetector.onTouchEvent(event)

                when (event.actionMasked) {
                    MotionEvent.ACTION_DOWN -> {
                        lastTouchX = event.x
                        lastTouchY = event.y
                        activePointerId = event.getPointerId(0)
                    }
                    MotionEvent.ACTION_MOVE -> {
                        if (!scaleGestureDetector.isInProgress) {
                            val pointerIndex = event.findPointerIndex(activePointerId)
                            if (pointerIndex >= 0) {
                                val x = event.getX(pointerIndex)
                                val y = event.getY(pointerIndex)
                                translateX += x - lastTouchX
                                translateY += y - lastTouchY
                                lastTouchX = x
                                lastTouchY = y
                                updateImageTransform()
                            }
                        }
                    }
                    MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                        activePointerId = MotionEvent.INVALID_POINTER_ID
                    }
                    MotionEvent.ACTION_POINTER_UP -> {
                        val pointerIndex = event.actionIndex
                        val pointerId = event.getPointerId(pointerIndex)
                        if (pointerId == activePointerId) {
                            val newPointerIndex = if (pointerIndex == 0) 1 else 0
                            lastTouchX = event.getX(newPointerIndex)
                            lastTouchY = event.getY(newPointerIndex)
                            activePointerId = event.getPointerId(newPointerIndex)
                        }
                    }
                }
                true
            }

            // Initialize matrix after layout
            mainImageView.post {
                // Center the image initially
                val drawable = mainImageView.drawable ?: return@post
                val dWidth = drawable.intrinsicWidth.toFloat()
                val dHeight = drawable.intrinsicHeight.toFloat()
                val vWidth = mainImageView.width.toFloat()
                val vHeight = mainImageView.height.toFloat()

                val scale = min(vWidth / dWidth, vHeight / dHeight)
                scaleFactor = scale

                translateX = (vWidth - dWidth * scale) / 2f
                translateY = (vHeight - dHeight * scale) / 2f

                imageMatrix.reset()
                imageMatrix.postScale(scale, scale)
                imageMatrix.postTranslate(translateX, translateY)
                mainImageView.imageMatrix = imageMatrix

                // Reset for proper pan/zoom from centered position
                translateX = 0f
                translateY = 0f
            }

            rootLayout.addView(mainImageView)
        } else {
            mainImageView = ImageView(this)  // Dummy to avoid uninitialized
            val placeholder = TextView(this).apply {
                text = getString(R.string.room_viewer_preview_unavailable)
                textSize = 16f
                setTextColor(Color.GRAY)
                gravity = Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    0, 1f
                )
            }
            rootLayout.addView(placeholder)
        }

        // Texture previews (thumbnails)
        val textureBar = HorizontalScrollView(this).apply {
            setBackgroundColor(Color.parseColor("#2A2A2A"))
            setPadding(8, 8, 8, 8)
        }

        val textureContainer = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        // Add texture thumbnails
        val textureFiles = listOf(
            "front_wall.png" to "Front",
            "floor.png" to "Floor",
            "ceiling.png" to "Ceiling",
            "left_wall.png" to "Left",
            "right_wall.png" to "Right"
        )

        for ((fileName, label) in textureFiles) {
            val file = File(roomFolder, fileName)
            if (file.exists()) {
                val container = LinearLayout(this).apply {
                    orientation = LinearLayout.VERTICAL
                    gravity = Gravity.CENTER
                    setPadding(8, 8, 8, 8)
                }

                val bitmap = BitmapFactory.decodeFile(file.absolutePath)
                val thumb = ImageView(this).apply {
                    setImageBitmap(bitmap)
                    scaleType = ImageView.ScaleType.CENTER_CROP
                    layoutParams = LinearLayout.LayoutParams(100, 100)
                }
                container.addView(thumb)

                val labelView = TextView(this).apply {
                    text = label
                    textSize = 10f
                    setTextColor(Color.LTGRAY)
                    gravity = Gravity.CENTER
                }
                container.addView(labelView)

                textureContainer.addView(container)
            }
        }

        textureBar.addView(textureContainer)
        rootLayout.addView(textureBar)

        // Info panel
        val infoPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#333333"))
            setPadding(24, 16, 24, 24)

            val infoTitle = TextView(this@RoomViewerActivity).apply {
                text = getString(R.string.room_viewer_created_success)
                textSize = 16f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.WHITE)
            }
            addView(infoTitle)

            val infoText = TextView(this@RoomViewerActivity).apply {
                text = getString(R.string.room_viewer_created_message)
                textSize = 14f
                setTextColor(Color.LTGRAY)
                setPadding(0, 8, 0, 0)
            }
            addView(infoText)

            // Read dimensions if available
            val dimensionsFile = File(roomFolder, "dimensions.txt")
            if (dimensionsFile.exists()) {
                val dims = dimensionsFile.readText()
                val dimsView = TextView(this@RoomViewerActivity).apply {
                    text = getString(R.string.room_viewer_dimensions, dims.replace("\n", ", "))
                    textSize = 12f
                    setTextColor(Color.GRAY)
                    setPadding(0, 8, 0, 0)
                }
                addView(dimsView)
            }
        }
        rootLayout.addView(infoPanel)

        setContentView(rootLayout)
    }
}
