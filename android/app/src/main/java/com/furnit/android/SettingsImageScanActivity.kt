package com.furnit.android

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Space
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.furnit.android.models.PhotoOrientation
import com.furnit.android.services.FurnitureFitManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class SettingsImageScanActivity : AppCompatActivity() {
    private lateinit var selectedImageView: ImageView
    private lateinit var overlayView: FurnitureFitOverlayView
    private lateinit var placeholderView: LinearLayout
    private lateinit var statusView: TextView
    private lateinit var chooseButton: TextView

    private val furnitureFitManager by lazy { FurnitureFitManager(this) }
    private var furnitureFitInitialized = false
    private var selectedBitmap: Bitmap? = null
    private var scanRequestId = 0

    private val imagePickerLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            if (uri != null) {
                loadImageFromUri(uri)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(createContentView())
        preloadModel()
    }

    override fun onDestroy() {
        if (furnitureFitInitialized) {
            furnitureFitManager.close()
        }
        super.onDestroy()
    }

    private fun createContentView(): View {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#F5F5F5"))
        }

        val topBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dpToPx(20), dpToPx(20), dpToPx(20), dpToPx(12))
        }
        val backButton = TextView(this).apply {
            text = "< ${getString(R.string.common_back)}"
            textSize = 16f
            setTextColor(Color.parseColor("#007AFF"))
            setOnClickListener { finish() }
        }
        val title = TextView(this).apply {
            text = getString(R.string.settings_image_scan)
            textSize = 20f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#333333"))
            gravity = Gravity.CENTER
        }
        topBar.addView(backButton)
        topBar.addView(Space(this), LinearLayout.LayoutParams(0, 0, 1f))
        topBar.addView(title)
        topBar.addView(Space(this), LinearLayout.LayoutParams(0, 0, 1f))
        root.addView(topBar)

        val body = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(16), 0, dpToPx(16), dpToPx(16))
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1f,
            )
        }

        val previewCard = FrameLayout(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1f,
            )
            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(18).toFloat()
                setColor(Color.parseColor("#1A1A1A"))
            }
            clipToOutline = true
        }

        selectedImageView = ImageView(this).apply {
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(Color.parseColor("#1A1A1A"))
            adjustViewBounds = true
            visibility = View.GONE
        }
        previewCard.addView(
            selectedImageView,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )

        overlayView = FurnitureFitOverlayView(this).apply {
            setBackgroundColor(Color.TRANSPARENT)
            setDetectionBoxVisibility(true)
            visibility = View.GONE
        }
        previewCard.addView(
            overlayView,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )

        placeholderView = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(dpToPx(24), dpToPx(24), dpToPx(24), dpToPx(24))
            setOnClickListener { openImagePicker() }
        }
        placeholderView.addView(
            TextView(this).apply {
                text = "\uD83D\uDDBC\uFE0F"
                textSize = 34f
                gravity = Gravity.CENTER
                setTextColor(Color.parseColor("#CCCCCC"))
            },
        )
        placeholderView.addView(
            TextView(this).apply {
                text = getString(R.string.settings_image_scan_tap_to_choose)
                textSize = 18f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
                setPadding(0, dpToPx(12), 0, dpToPx(8))
            },
        )
        placeholderView.addView(
            TextView(this).apply {
                text = getString(R.string.settings_image_scan_tap_to_choose_subtitle)
                textSize = 14f
                setTextColor(Color.parseColor("#CCCCCC"))
                gravity = Gravity.CENTER
            },
        )
        previewCard.addView(
            placeholderView,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )

        statusView = TextView(this).apply {
            text = getString(R.string.settings_image_scan_preparing_model)
            textSize = 14f
            setTextColor(Color.WHITE)
            setPadding(dpToPx(14), dpToPx(10), dpToPx(14), dpToPx(10))
            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(18).toFloat()
                setColor(Color.argb(170, 0, 0, 0))
            }
            gravity = Gravity.CENTER
        }
        previewCard.addView(
            statusView,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.CENTER,
            ),
        )

        body.addView(previewCard)

        val footer = ScrollView(this).apply {
            overScrollMode = View.OVER_SCROLL_NEVER
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply {
                topMargin = dpToPx(16)
            }
        }
        val footerContent = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(4), 0, dpToPx(4), 0)
        }
        chooseButton = TextView(this).apply {
            text = getString(R.string.settings_image_scan_tap_to_choose)
            textSize = 16f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#007AFF"))
            setPadding(0, 0, 0, dpToPx(8))
            setOnClickListener { openImagePicker() }
        }
        footerContent.addView(chooseButton)
        footerContent.addView(
            TextView(this).apply {
                text = getString(R.string.settings_image_scan_footnote)
                textSize = 13f
                setTextColor(Color.parseColor("#666666"))
            },
        )
        footer.addView(footerContent)
        body.addView(footer)

        root.addView(body)
        return root
    }

    private fun preloadModel() {
        statusView.visibility = View.VISIBLE
        statusView.text = getString(R.string.settings_image_scan_preparing_model)
        lifecycleScope.launch {
            val initialized = withContext(Dispatchers.IO) {
                furnitureFitManager.initializeAuto()
            }
            furnitureFitInitialized = initialized
            if (isDestroyed) return@launch
            if (selectedBitmap == null) {
                statusView.visibility = if (initialized) View.GONE else View.VISIBLE
            }
            if (!initialized) {
                statusView.text = getString(R.string.yoloe_model_unavailable)
            }
        }
    }

    private fun openImagePicker() {
        imagePickerLauncher.launch("image/*")
    }

    private fun loadImageFromUri(uri: Uri) {
        lifecycleScope.launch {
            statusView.visibility = View.VISIBLE
            statusView.text = getString(R.string.settings_image_scan_loading_photo)
            val bitmap = withContext(Dispatchers.IO) {
                PhotoOrientation.loadBitmapApplyingExif(this@SettingsImageScanActivity, uri)
            }
            if (bitmap == null) {
                statusView.text = getString(R.string.settings_image_scan_load_failed)
                Toast.makeText(
                    this@SettingsImageScanActivity,
                    getString(R.string.settings_image_scan_load_failed),
                    Toast.LENGTH_SHORT,
                ).show()
                return@launch
            }
            selectedBitmap = bitmap
            selectedImageView.setImageBitmap(bitmap)
            selectedImageView.visibility = View.VISIBLE
            overlayView.visibility = View.VISIBLE
            placeholderView.visibility = View.GONE
            chooseButton.text = getString(R.string.settings_image_scan_change_photo)
            startSingleImageOverlayScan(bitmap)
        }
    }

    private fun startSingleImageOverlayScan(bitmap: Bitmap) {
        val requestId = ++scanRequestId
        overlayView.setMaskAndDetections(null, emptyList(), 640)
        overlayView.resetTransform()
        statusView.text = getString(R.string.single_image_scan_scanning)
        statusView.visibility = View.VISIBLE
        lifecycleScope.launch {
            val initialized = if (furnitureFitInitialized) {
                true
            } else {
                withContext(Dispatchers.IO) {
                    furnitureFitManager.initializeAuto()
                }.also { success ->
                    furnitureFitInitialized = success
                }
            }

            if (requestId != scanRequestId || isDestroyed) return@launch

            if (!initialized) {
                statusView.text = getString(R.string.yoloe_model_unavailable)
                return@launch
            }

            furnitureFitManager.segmentWithDetectionsAsync(bitmap) { result ->
                if (requestId != scanRequestId || isDestroyed) return@segmentWithDetectionsAsync
                runOnUiThread {
                    if (result == null) {
                        overlayView.setMaskAndDetections(null, emptyList(), 640)
                        statusView.text = getString(R.string.single_image_scan_no_overlay)
                        return@runOnUiThread
                    }
                    overlayView.setMaskAndDetections(result.mask, result.detections, result.inputSize)
                    statusView.text = if (result.detections.isEmpty()) {
                        getString(R.string.single_image_scan_no_objects)
                    } else {
                        getString(R.string.single_image_scan_overlay_ready, result.detections.first().label)
                    }
                    statusView.postDelayed({
                        if (requestId == scanRequestId && !isDestroyed) {
                            statusView.visibility = View.GONE
                        }
                    }, 1800L)
                }
            }
        }
    }

    private fun dpToPx(dp: Int): Int {
        return (dp * resources.displayMetrics.density).toInt()
    }
}
