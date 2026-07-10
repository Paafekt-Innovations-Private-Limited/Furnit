package com.furnit.android

import android.content.res.ColorStateList
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Typeface
import android.os.Bundle
import com.furnit.android.theme.PaafektColors
import com.furnit.android.theme.PaafektDrawables
import com.furnit.android.theme.PaafektHintController
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
    private lateinit var rootLayout: FrameLayout
    private lateinit var hintController: PaafektHintController

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

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

        roomFolder.listFiles()?.forEach { file ->
            LogUtil.d("RoomViewer", "  File: ${file.name}")
        }

        scaleGestureDetector = ScaleGestureDetector(this, ScaleListener())
        setupUI(roomFolder)
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private inner class ScaleListener : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector: ScaleGestureDetector): Boolean {
            scaleFactor *= detector.scaleFactor
            scaleFactor = max(0.5f, min(scaleFactor, 5.0f))
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
        rootLayout = FrameLayout(this).apply {
            setBackgroundColor(PaafektColors.background)
        }

        val contentColumn = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            )
        }

        val frontWallFile = File(roomFolder, "front_wall.png")
        if (frontWallFile.exists()) {
            val bitmap = BitmapFactory.decodeFile(frontWallFile.absolutePath)
            mainImageView = ImageView(this).apply {
                setImageBitmap(bitmap)
                scaleType = ImageView.ScaleType.MATRIX
                setBackgroundColor(Color.BLACK)
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    0,
                    1f,
                )
            }

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

            mainImageView.post {
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

                translateX = 0f
                translateY = 0f
            }

            contentColumn.addView(mainImageView)
        } else {
            mainImageView = ImageView(this)
            val placeholder = TextView(this).apply {
                text = getString(R.string.room_viewer_preview_unavailable)
                textSize = 16f
                setTextColor(PaafektColors.textSecondary)
                gravity = Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    0,
                    1f,
                )
            }
            contentColumn.addView(placeholder)
        }

        val textureBar = HorizontalScrollView(this).apply {
            setBackgroundColor(PaafektColors.surface)
            setPadding(dp(8), dp(8), dp(8), dp(8))
        }

        val textureContainer = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        val textureFiles = listOf(
            "front_wall.png" to "Front",
            "floor.png" to "Floor",
            "ceiling.png" to "Ceiling",
            "left_wall.png" to "Left",
            "right_wall.png" to "Right",
        )

        for ((fileName, label) in textureFiles) {
            val file = File(roomFolder, fileName)
            if (file.exists()) {
                val container = LinearLayout(this).apply {
                    orientation = LinearLayout.VERTICAL
                    gravity = Gravity.CENTER
                    setPadding(dp(8), dp(8), dp(8), dp(8))
                }

                val bitmap = BitmapFactory.decodeFile(file.absolutePath)
                val thumb = ImageView(this).apply {
                    setImageBitmap(bitmap)
                    scaleType = ImageView.ScaleType.CENTER_CROP
                    layoutParams = LinearLayout.LayoutParams(dp(100), dp(100))
                }
                container.addView(thumb)

                val labelView = TextView(this).apply {
                    text = label
                    textSize = 10f
                    setTextColor(PaafektColors.textSecondary)
                    gravity = Gravity.CENTER
                }
                container.addView(labelView)

                textureContainer.addView(container)
            }
        }

        textureBar.addView(textureContainer)
        contentColumn.addView(textureBar)

        val infoPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(PaafektColors.surfaceHi)
            setPadding(dp(24), dp(16), dp(24), dp(24))

            val infoTitle = TextView(this@RoomViewerActivity).apply {
                text = getString(R.string.room_viewer_created_success)
                textSize = 16f
                setTypeface(null, Typeface.BOLD)
                setTextColor(PaafektColors.textPrimary)
            }
            addView(infoTitle)

            val infoText = TextView(this@RoomViewerActivity).apply {
                text = getString(R.string.room_viewer_created_message)
                textSize = 14f
                setTextColor(PaafektColors.textSecondary)
                setPadding(0, dp(8), 0, 0)
            }
            addView(infoText)

            val dimensionsFile = File(roomFolder, "dimensions.txt")
            if (dimensionsFile.exists()) {
                val heightMeters = dimensionsFile.readText()
                    .lineSequence()
                    .map { it.trim() }
                    .firstOrNull { it.startsWith("height=") }
                    ?.substringAfter("height=")
                    ?.toFloatOrNull()
                val dimsView = TextView(this@RoomViewerActivity).apply {
                    text = if (heightMeters != null && heightMeters > 0f) {
                        getString(R.string.approximate_room_height, heightMeters)
                    } else {
                        getString(R.string.room_viewer_dimensions, dimensionsFile.readText().replace("\n", ", "))
                    }
                    textSize = 12f
                    setTextColor(PaafektColors.textSecondary)
                    setPadding(0, dp(8), 0, 0)
                }
                addView(dimsView)
            }
        }
        contentColumn.addView(infoPanel)

        rootLayout.addView(contentColumn)

        val topBar = FrameLayout(this).apply {
            setPadding(dp(16), dp(48), dp(16), 0)
            elevation = 40f
        }

        val backBtn = TextView(this).apply {
            text = getString(R.string.photo_room_back)
            textSize = 20f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
            background = PaafektDrawables.toolbarCircle()
            setOnClickListener { finish() }
        }
        topBar.addView(
            backBtn,
            FrameLayout.LayoutParams(dp(36), dp(36)).apply {
                gravity = Gravity.START or Gravity.TOP
            },
        )

        val titleCapsule = TextView(this).apply {
            text = getString(R.string.room_viewer_your_room)
            textSize = 14f
            setTypeface(null, Typeface.BOLD)
            setTextColor(PaafektColors.textPrimary)
            gravity = Gravity.CENTER
            setPadding(dp(16), dp(8), dp(16), dp(8))
            background = PaafektDrawables.toolbarCapsule()
        }
        topBar.addView(
            titleCapsule,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply { gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL },
        )

        val saveBtn = TextView(this).apply {
            text = getString(R.string.common_save)
            textSize = 14f
            setTypeface(null, Typeface.BOLD)
            setTextColor(PaafektColors.accentText)
            gravity = Gravity.CENTER
            setPadding(dp(12), dp(8), dp(12), dp(8))
            background = PaafektDrawables.primaryButton()
            setOnClickListener {
                Toast.makeText(this@RoomViewerActivity, getString(R.string.room_viewer_saved_toast), Toast.LENGTH_SHORT).show()
                finish()
            }
        }
        topBar.addView(
            saveBtn,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply { gravity = Gravity.END or Gravity.TOP },
        )

        rootLayout.addView(
            topBar,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply { gravity = Gravity.TOP },
        )

        hintController = PaafektHintController(rootLayout)
        setContentView(rootLayout)

        rootLayout.post {
            hintController.show(
                this,
                R.drawable.ic_gesture_pinch,
                R.string.room_viewer_pinch_hint,
            )
        }
    }
}
