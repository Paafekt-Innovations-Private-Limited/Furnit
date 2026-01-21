package com.furnit.android

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.furnit.android.models.RoomStructure
import com.furnit.android.services.SharpService
import java.io.InputStream

/**
 * SinglePhotoRoomActivity - Image picker with Manual/AI room creation options
 * (Matches Swift's SinglePhotoRoomView)
 *
 * Flow:
 * 1. User picks a photo from gallery
 * 2. Shows preview with two options: Manual Setup or AI Room (Sharp)
 * 3. Manual Setup: boundary adjustment for room creation
 * 4. AI Room: Sharp-based 3D generation
 */
class SinglePhotoRoomActivity : AppCompatActivity() {

    private lateinit var rootLayout: FrameLayout
    private lateinit var initialView: LinearLayout
    private lateinit var methodPickerView: LinearLayout
    private lateinit var progressOverlay: FrameLayout
    private lateinit var progressBar: ProgressBar
    private lateinit var progressText: TextView
    private lateinit var progressPercent: TextView
    private lateinit var selectedImageView: ImageView
    private var selectedBitmap: Bitmap? = null
    private var selectedImageUri: Uri? = null

    private val imagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            Log.d("SinglePhotoRoom", "Image selected: $uri")
            selectedImageUri = uri
            loadImageFromUri(uri)
        } else {
            Log.d("SinglePhotoRoom", "No image selected")
        }
    }

    private val boundaryActivityLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            Log.d("SinglePhotoRoom", "Boundary adjustment completed")
            // TODO: Get boundaries from result and process room
            // val boundaries = result.data?.getSerializableExtra(RoomBoundaryActivity.RESULT_BOUNDARIES) as? RoomStructure
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        rootLayout = FrameLayout(this)
        rootLayout.setBackgroundColor(Color.parseColor("#F5F5F5"))

        // Initial view - photo selection
        initialView = createInitialView()
        rootLayout.addView(initialView)

        // Method picker view - hidden initially
        methodPickerView = createMethodPickerView()
        methodPickerView.visibility = View.GONE
        rootLayout.addView(methodPickerView)

        // Progress overlay - hidden initially
        progressOverlay = createProgressOverlay()
        progressOverlay.visibility = View.GONE
        rootLayout.addView(progressOverlay)

        setContentView(rootLayout)
    }

    private fun createInitialView(): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(48, 80, 48, 48)

            // Back button
            val backBtn = TextView(this@SinglePhotoRoomActivity).apply {
                text = "< Back"
                textSize = 16f
                setTextColor(Color.parseColor("#007AFF"))
                setPadding(0, 0, 0, 32)
                setOnClickListener { finish() }
            }
            addView(backBtn, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ))

            // Title
            val title = TextView(this@SinglePhotoRoomActivity).apply {
                text = "Create 3D Room"
                textSize = 24f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.parseColor("#333333"))
                gravity = Gravity.CENTER
            }
            addView(title)

            // Subtitle
            val subtitle = TextView(this@SinglePhotoRoomActivity).apply {
                text = "Select a photo of your room"
                textSize = 16f
                setTextColor(Color.parseColor("#666666"))
                gravity = Gravity.CENTER
                setPadding(0, 16, 0, 48)
            }
            addView(subtitle)

            // Photo selection button
            val selectPhotoBtn = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setPadding(48, 48, 48, 48)
                setBackgroundColor(Color.parseColor("#E8F5E9"))

                val icon = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "\uD83D\uDDBC️" // Frame icon
                    textSize = 48f
                    gravity = Gravity.CENTER
                }
                addView(icon)

                val btnText = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "Select Photo"
                    textSize = 18f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(Color.parseColor("#4CAF50"))
                    gravity = Gravity.CENTER
                    setPadding(0, 16, 0, 0)
                }
                addView(btnText)

                val btnHint = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "From your photo library"
                    textSize = 14f
                    setTextColor(Color.parseColor("#666666"))
                    gravity = Gravity.CENTER
                    setPadding(0, 8, 0, 0)
                }
                addView(btnHint)

                setOnClickListener {
                    openImagePicker()
                }
            }
            addView(selectPhotoBtn, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 0, 0, 32) })

            // Warning
            val warning = TextView(this@SinglePhotoRoomActivity).apply {
                text = "⚠️ Do not use screenshots - use actual photos"
                textSize = 14f
                setTextColor(Color.parseColor("#F44336"))
                gravity = Gravity.CENTER
            }
            addView(warning)
        }
    }

    private fun createMethodPickerView(): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            setPadding(32, 48, 32, 32)

            // Back button
            val backBtn = TextView(this@SinglePhotoRoomActivity).apply {
                text = "< Back"
                textSize = 16f
                setTextColor(Color.parseColor("#007AFF"))
                setPadding(0, 0, 0, 16)
                setOnClickListener {
                    showInitialView()
                }
            }
            addView(backBtn)

            // Image preview
            selectedImageView = ImageView(this@SinglePhotoRoomActivity).apply {
                scaleType = ImageView.ScaleType.CENTER_CROP
                setBackgroundColor(Color.parseColor("#E0E0E0"))
            }
            addView(selectedImageView, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                400
            ).apply { setMargins(0, 0, 0, 24) })

            // Title
            val title = TextView(this@SinglePhotoRoomActivity).apply {
                text = "How would you like to create your room?"
                textSize = 18f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.parseColor("#333333"))
                gravity = Gravity.CENTER
            }
            addView(title)

            val subtitle = TextView(this@SinglePhotoRoomActivity).apply {
                text = "Tap an option below"
                textSize = 14f
                setTextColor(Color.parseColor("#666666"))
                gravity = Gravity.CENTER
                setPadding(0, 8, 0, 24)
            }
            addView(subtitle)

            // AI Room option (Sharp)
            val aiOption = createOptionCard(
                icon = "\uD83E\uDE84", // Magic wand
                title = "AI Room",
                subtitle = "AI-powered 3D generation",
                bgColor = "#F3E5F5",
                borderColor = "#9C27B0"
            ) {
                onAIRoomSelected()
            }
            addView(aiOption)

            // Manual Setup option
            val manualOption = createOptionCard(
                icon = "\uD83D\uDCCF", // Ruler
                title = "Manual Setup",
                subtitle = "Adjust room boundaries manually",
                bgColor = "#FFF3E0",
                borderColor = "#FF9800"
            ) {
                onManualSetupSelected()
            }
            addView(manualOption)

            // Change photo button
            val changePhotoBtn = TextView(this@SinglePhotoRoomActivity).apply {
                text = "Choose Different Photo"
                textSize = 14f
                setTextColor(Color.parseColor("#666666"))
                gravity = Gravity.CENTER
                setPadding(0, 32, 0, 0)
                setOnClickListener {
                    openImagePicker()
                }
            }
            addView(changePhotoBtn)
        }
    }

    private fun createOptionCard(
        icon: String,
        title: String,
        subtitle: String,
        bgColor: String,
        borderColor: String,
        onClick: () -> Unit
    ): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(Color.parseColor(bgColor))
            setPadding(24, 24, 24, 24)
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 0, 0, 16) }

            // Icon
            val iconView = TextView(this@SinglePhotoRoomActivity).apply {
                text = icon
                textSize = 32f
            }
            addView(iconView, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 0, 24, 0) })

            // Text container
            val textContainer = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)

                val titleView = TextView(this@SinglePhotoRoomActivity).apply {
                    text = title
                    textSize = 16f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(Color.parseColor("#333333"))
                }
                addView(titleView)

                val subtitleView = TextView(this@SinglePhotoRoomActivity).apply {
                    text = subtitle
                    textSize = 12f
                    setTextColor(Color.parseColor("#666666"))
                }
                addView(subtitleView)
            }
            addView(textContainer)

            // Chevron
            val chevron = TextView(this@SinglePhotoRoomActivity).apply {
                text = ">"
                textSize = 18f
                setTextColor(Color.parseColor("#999999"))
            }
            addView(chevron)

            setOnClickListener { onClick() }
        }
    }

    private fun openImagePicker() {
        Log.d("SinglePhotoRoom", "Opening image picker")
        imagePickerLauncher.launch("image/*")
    }

    private fun loadImageFromUri(uri: Uri) {
        try {
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()

            if (bitmap != null) {
                selectedBitmap = bitmap
                selectedImageView.setImageBitmap(bitmap)
                showMethodPicker()
                Log.d("SinglePhotoRoom", "Image loaded: ${bitmap.width}x${bitmap.height}")
            } else {
                Log.e("SinglePhotoRoom", "Failed to decode image")
                Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e("SinglePhotoRoom", "Error loading image", e)
            Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun showMethodPicker() {
        initialView.visibility = View.GONE
        methodPickerView.visibility = View.VISIBLE
    }

    private fun onAIRoomSelected() {
        Log.d("SinglePhotoRoom", "AI Room (Sharp) selected")
        val bitmap = selectedBitmap
        if (bitmap == null) {
            Toast.makeText(this, "No image selected", Toast.LENGTH_SHORT).show()
            return
        }

        // Show progress overlay
        showProgressOverlay()

        // Start Sharp generation
        val sharpService = SharpService(this)
        sharpService.generateGaussians(bitmap, object : SharpService.ProgressCallback {
            override fun onProgress(progress: Float, message: String) {
                runOnUiThread {
                    updateProgressOverlay(progress, message)
                }
            }

            override fun onComplete(result: SharpService.GenerationResult) {
                runOnUiThread {
                    hideProgressOverlay()
                    Log.d("SinglePhotoRoom", "AI Room generated: ${result.plyFile.absolutePath}")

                    // Open SharpRoomActivity with the generated PLY
                    val intent = Intent(this@SinglePhotoRoomActivity, SharpRoomActivity::class.java).apply {
                        putExtra(SharpRoomActivity.EXTRA_PLY_PATH, result.classicPlyFile.absolutePath)
                        putExtra(SharpRoomActivity.EXTRA_ROOM_FOLDER, result.plyFile.parentFile?.absolutePath)
                        putExtra(SharpRoomActivity.EXTRA_ROOM_WIDTH, result.roomWidth)
                        putExtra(SharpRoomActivity.EXTRA_ROOM_HEIGHT, result.roomHeight)
                        putExtra(SharpRoomActivity.EXTRA_ROOM_DEPTH, result.roomDepth)
                        putExtra(SharpRoomActivity.EXTRA_ALLOW_SAVE, true)
                    }
                    startActivity(intent)
                }
            }

            override fun onError(message: String) {
                runOnUiThread {
                    hideProgressOverlay()
                    Toast.makeText(this@SinglePhotoRoomActivity, message, Toast.LENGTH_LONG).show()
                }
            }
        })
    }

    private fun onManualSetupSelected() {
        Log.d("SinglePhotoRoom", "Manual Setup selected")
        val uri = selectedImageUri
        if (uri == null) {
            Toast.makeText(this, "No image selected", Toast.LENGTH_SHORT).show()
            return
        }

        val intent = Intent(this, RoomBoundaryActivity::class.java).apply {
            putExtra(RoomBoundaryActivity.EXTRA_IMAGE_URI, uri.toString())
        }
        boundaryActivityLauncher.launch(intent)
    }

    private fun showInitialView() {
        methodPickerView.visibility = View.GONE
        initialView.visibility = View.VISIBLE
        selectedBitmap = null
        selectedImageUri = null
    }

    private fun createProgressOverlay(): FrameLayout {
        return FrameLayout(this).apply {
            setBackgroundColor(Color.parseColor("#CC000000"))
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )

            val content = LinearLayout(this@SinglePhotoRoomActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setPadding(64, 48, 64, 48)
                setBackgroundColor(Color.parseColor("#FFFFFF"))

                // Icon/animation container
                val iconContainer = FrameLayout(this@SinglePhotoRoomActivity).apply {
                    val circleSize = 120
                    layoutParams = LinearLayout.LayoutParams(circleSize, circleSize).apply {
                        gravity = Gravity.CENTER
                    }

                    // Background circle
                    val bgCircle = View(this@SinglePhotoRoomActivity).apply {
                        setBackgroundColor(Color.parseColor("#E1BEE7"))
                    }
                    addView(bgCircle, FrameLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    ))

                    // Magic wand icon
                    val icon = TextView(this@SinglePhotoRoomActivity).apply {
                        text = "\uD83E\uDE84"
                        textSize = 40f
                        gravity = Gravity.CENTER
                    }
                    addView(icon, FrameLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    ).apply { gravity = Gravity.CENTER })
                }
                addView(iconContainer)

                // Progress text
                progressText = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "Creating your 3D room..."
                    textSize = 18f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(Color.parseColor("#333333"))
                    gravity = Gravity.CENTER
                    setPadding(0, 32, 0, 16)
                }
                addView(progressText)

                // Progress bar
                progressBar = ProgressBar(this@SinglePhotoRoomActivity, null, android.R.attr.progressBarStyleHorizontal).apply {
                    max = 100
                    progress = 0
                    layoutParams = LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT
                    ).apply { setMargins(0, 0, 0, 16) }
                }
                addView(progressBar)

                // Percentage text
                progressPercent = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "0%"
                    textSize = 24f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(Color.parseColor("#9C27B0"))
                    gravity = Gravity.CENTER
                }
                addView(progressPercent)

                // Subtitle
                val subtitle = TextView(this@SinglePhotoRoomActivity).apply {
                    text = "AI-powered room generation"
                    textSize = 12f
                    setTextColor(Color.parseColor("#999999"))
                    gravity = Gravity.CENTER
                    setPadding(0, 16, 0, 0)
                }
                addView(subtitle)
            }

            addView(content, FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { gravity = Gravity.CENTER })
        }
    }

    private fun showProgressOverlay() {
        progressOverlay.visibility = View.VISIBLE
        progressBar.progress = 0
        progressPercent.text = "0%"
        progressText.text = "Preparing..."
    }

    private fun hideProgressOverlay() {
        progressOverlay.visibility = View.GONE
    }

    private fun updateProgressOverlay(progress: Float, message: String) {
        val percent = (progress * 100).toInt()
        progressBar.progress = percent
        progressPercent.text = "$percent%"
        progressText.text = message
    }
}
