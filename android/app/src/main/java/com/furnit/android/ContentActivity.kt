package com.furnit.android

import android.content.Intent
import android.graphics.BitmapFactory
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import com.furnit.android.utils.CrashReporter
import com.furnit.android.utils.LogUtil
import androidx.appcompat.app.AppCompatActivity
import android.widget.LinearLayout
import android.view.Gravity
import android.widget.TextView
import android.widget.ImageView
import android.view.ViewGroup
import android.graphics.Color
import android.graphics.Typeface
import android.view.View
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ScrollView
import android.widget.Toast
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.furnit.android.services.SharpGenerationUiState
import com.furnit.android.services.SharpService
import androidx.appcompat.app.AlertDialog
import com.furnit.android.auth.AuthenticationManager
import com.furnit.android.auth.LoginActivity
import com.furnit.android.models.Model
import com.furnit.android.models.ModelManager
import java.io.File

class ContentActivity : AppCompatActivity() {
    private lateinit var modelManager: ModelManager
    private lateinit var authManager: AuthenticationManager
    private lateinit var roomsContainer: LinearLayout
    private lateinit var statsText: TextView
    private lateinit var totalSizeText: TextView
    private lateinit var sharpGlobalProgressBar: FrameLayout
    private lateinit var sharpGlobalProgressLabel: TextView

    // Colors matching iOS dark theme
    private val backgroundColor = Color.parseColor("#1C1C1E")
    private val cardBackgroundColor = Color.parseColor("#2C2C2E")
    private val primaryTextColor = Color.WHITE
    private val secondaryTextColor = Color.parseColor("#8E8E93")
    private val accentGreen = Color.parseColor("#34C759")
    private val accentPurple = Color.parseColor("#AF52DE")
    private val dividerColor = Color.parseColor("#3A3A3C")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        authManager = AuthenticationManager.getInstance(this)

        // Check authentication
        if (!authManager.isAuthenticated) {
            navigateToLogin()
            return
        }

        modelManager = ModelManager(this)
        setupUI()

        // Python warmup disabled (needs Chaquopy + ARM PyTorch wheels)
    }

    override fun onResume() {
        super.onResume()

        // Check authentication on resume
        if (!authManager.isAuthenticated) {
            navigateToLogin()
            return
        }

        // Refresh models when returning to this activity
        modelManager.refresh()
        refreshRoomsList()
        syncSharpGlobalProgressBarFromState()
    }

    override fun onDestroy() {
        SharpGenerationUiState.setListener(null)
        super.onDestroy()
    }

    private fun navigateToLogin() {
        val intent = Intent(this, LoginActivity::class.java)
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        startActivity(intent)
        finish()
    }

    private fun dpToPx(dp: Int): Int {
        return (dp * resources.displayMetrics.density).toInt()
    }

    private fun setupUI() {
        val scrollView = ScrollView(this).apply {
            setBackgroundColor(backgroundColor)
        }

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dpToPx(16), dpToPx(16), dpToPx(16), dpToPx(16))
        }

        // Top bar with icons
        val topBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dpToPx(8), 0, dpToPx(16))
        }

        // Create Room entry (left): icon + helper text
        val createRoomEntry = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setOnClickListener {
                startActivity(Intent(this@ContentActivity, SinglePhotoRoomActivity::class.java))
            }
        }
        val createRoomIcon = createIconButton("\uD83D\uDDBC") // Image icon
        createRoomEntry.addView(createRoomIcon)
        val createRoomHelper = TextView(this).apply {
            text = getString(R.string.home_create_room_helper)
            textSize = 14f
            setTextColor(primaryTextColor)
            setPadding(dpToPx(8), 0, 0, 0)
        }
        createRoomEntry.addView(createRoomHelper)
        topBar.addView(createRoomEntry)

        // Spacer
        val spacer = View(this).apply {
            layoutParams = LinearLayout.LayoutParams(0, 1, 1f)
        }
        topBar.addView(spacer)

        // Help icon button - opens FAQ/Help Activity
        val helpIcon = createIconButton("?")
        helpIcon.setOnClickListener {
            startActivity(Intent(this@ContentActivity, HelpActivity::class.java))
        }
        topBar.addView(helpIcon)

        // Settings icon button
        val settingsIcon = createIconButton("\u2699") // Gear icon
        settingsIcon.setOnClickListener {
            startActivity(Intent(this@ContentActivity, SettingsActivity::class.java))
        }
        val settingsParams = LinearLayout.LayoutParams(dpToPx(44), dpToPx(44))
        settingsParams.setMargins(dpToPx(8), 0, 0, 0)
        settingsIcon.layoutParams = settingsParams
        topBar.addView(settingsIcon)

        layout.addView(topBar)

        // Stats row
        val statsRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 0, 0, dpToPx(8))
        }

        statsText = TextView(this).apply {
            text = getString(R.string.home_rooms_remaining_short, 0, 1000)
            textSize = 14f
            setTextColor(primaryTextColor)
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        statsRow.addView(statsText)

        val totalLabel = TextView(this).apply {
            text = getString(R.string.home_total)
            textSize = 12f
            setTextColor(secondaryTextColor)
            setPadding(0, 0, dpToPx(8), 0)
        }
        statsRow.addView(totalLabel)

        totalSizeText = TextView(this).apply {
            text = getString(R.string.home_zero_mb)
            textSize = 14f
            setTextColor(primaryTextColor)
            setTypeface(null, Typeface.BOLD)
        }
        statsRow.addView(totalSizeText)

        layout.addView(statsRow)

        // Swipe hint
        val hintRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dpToPx(8), 0, dpToPx(16))
        }

        val bulbIcon = TextView(this).apply {
            text = "\uD83D\uDCA1" // Lightbulb emoji
            textSize = 14f
            setPadding(0, 0, dpToPx(8), 0)
        }
        hintRow.addView(bulbIcon)

        val hintText = TextView(this).apply {
            text = getString(R.string.home_long_press_hint)
            textSize = 14f
            setTextColor(secondaryTextColor)
        }
        hintRow.addView(hintText)

        layout.addView(hintRow)

        // Divider
        layout.addView(createDivider())

        // Rooms container (will be refreshed)
        roomsContainer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        layout.addView(roomsContainer)

        scrollView.addView(layout)
        setContentView(scrollView)

        val contentRoot = findViewById<FrameLayout>(android.R.id.content)
        sharpGlobalProgressBar = createSharpGlobalProgressBar()
        contentRoot.addView(
            sharpGlobalProgressBar,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM,
            ),
        )
        SharpGenerationUiState.setListener { syncSharpGlobalProgressBarFromState() }

        // Initial load
        refreshRoomsList()
    }

    private fun createSharpGlobalProgressBar(): FrameLayout {
        val density = resources.displayMetrics.density
        return FrameLayout(this).apply {
            visibility = View.GONE
            setBackgroundColor(Color.parseColor("#5E35B1"))
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.LOLLIPOP) {
                elevation = 28f * density
            }
            val row = LinearLayout(this@ContentActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(dpToPx(14), dpToPx(10), dpToPx(14), dpToPx(14))
            }
            ViewCompat.setOnApplyWindowInsetsListener(this) { _, insets ->
                val nav = insets.getInsets(WindowInsetsCompat.Type.navigationBars())
                row.setPadding(dpToPx(14), dpToPx(10), dpToPx(14), dpToPx(14) + nav.bottom)
                insets
            }
            sharpGlobalProgressLabel = TextView(this@ContentActivity).apply {
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                textSize = 13f
                setTextColor(Color.WHITE)
                maxLines = 2
            }
            row.addView(sharpGlobalProgressLabel)
            val stopGlobal = TextView(this@ContentActivity).apply {
                text = "⏹"
                textSize = 11f
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
                setPadding(dpToPx(8), dpToPx(4), dpToPx(8), dpToPx(4))
                background = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(4).toFloat()
                    setColor(Color.parseColor("#E53935"))
                }
                contentDescription = getString(R.string.single_photo_ai_stop)
                setOnClickListener { onSharpGlobalStopClicked() }
            }
            row.addView(stopGlobal)
            addView(row)
        }
    }

    private fun syncSharpGlobalProgressBarFromState() {
        if (!::sharpGlobalProgressBar.isInitialized) return
        val s = SharpGenerationUiState
        if (!s.isGenerating) {
            sharpGlobalProgressBar.visibility = View.GONE
            return
        }
        sharpGlobalProgressBar.visibility = View.VISIBLE
        sharpGlobalProgressLabel.text = s.statusLine
        (sharpGlobalProgressBar.parent as? ViewGroup)?.bringChildToFront(sharpGlobalProgressBar)
        ViewCompat.requestApplyInsets(sharpGlobalProgressBar)
    }

    private fun onSharpGlobalStopClicked() {
        SharpService.getInstance(this).cancelGeneration()
        SharpGenerationUiState.clear()
        syncSharpGlobalProgressBarFromState()
        Toast.makeText(this, getString(R.string.home_sharp_generation_stopped), Toast.LENGTH_SHORT).show()
    }

    private fun createIconButton(icon: String): TextView {
        return TextView(this).apply {
            text = icon
            textSize = 20f
            setTextColor(primaryTextColor)
            gravity = Gravity.CENTER
            val bg = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(12).toFloat()
                setColor(cardBackgroundColor)
            }
            background = bg
            val params = LinearLayout.LayoutParams(dpToPx(44), dpToPx(44))
            layoutParams = params
        }
    }

    private fun createDivider(): View {
        return View(this).apply {
            setBackgroundColor(dividerColor)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dpToPx(1)
            )
        }
    }


    private fun refreshRoomsList() {
        roomsContainer.removeAllViews()

        val models = modelManager.listModels()
        LogUtil.d("ContentActivity", "Loaded ${models.size} models:")
        models.forEach { model ->
            LogUtil.d("ContentActivity", "  - ${model.name} (id=${model.id}, isUserCreated=${model.isUserCreated}, path=${model.assetPath})")
        }

        // Update stats
        val roomCount = models.size
        val remaining = 1000 - roomCount
        statsText.text = getString(R.string.home_rooms_remaining_short, remaining, 1000)

        // Calculate total size
        var totalBytes = 0L
        models.forEach { model ->
            try {
                val file = File(model.assetPath)
                if (file.exists()) {
                    totalBytes += if (file.isDirectory) {
                        file.walkTopDown().filter { it.isFile }.map { it.length() }.sum()
                    } else {
                        file.length()
                    }
                }
            } catch (e: Exception) {
                // Ignore
            }
        }
        val totalMB = totalBytes / (1024.0 * 1024.0)
        totalSizeText.text = if (totalMB >= 1024) {
            String.format("%.2f GB", totalMB / 1024.0)
        } else {
            String.format("%.1f MB", totalMB)
        }

        if (models.isEmpty()) {
            // Empty state
            val emptyText = TextView(this).apply {
                text = getString(R.string.home_no_rooms_yet)
                textSize = 16f
                setTextColor(secondaryTextColor)
                gravity = Gravity.CENTER
                setPadding(0, dpToPx(60), 0, dpToPx(60))
            }
            roomsContainer.addView(emptyText)
        } else {
            // Room cards
            for (model in models) {
                val card = createRoomCard(model)
                roomsContainer.addView(card)
                roomsContainer.addView(createDivider())
            }
        }
    }

    private fun createRoomCard(model: Model): View {
        val card = LinearLayout(this)
        card.orientation = LinearLayout.HORIZONTAL
        card.setBackgroundColor(backgroundColor)
        card.setPadding(0, dpToPx(12), 0, dpToPx(12))
        card.gravity = Gravity.CENTER_VERTICAL

        val params = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )
        card.layoutParams = params

        // Icon (purple grid for 3D Room, green box for 3D Model)
        val iconContainer = LinearLayout(this).apply {
            gravity = Gravity.CENTER
            val bg = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(8).toFloat()
                setColor(if (model.isUserCreated) cardBackgroundColor else Color.parseColor("#1A3D1A"))
            }
            background = bg
            val iconParams = LinearLayout.LayoutParams(dpToPx(44), dpToPx(44))
            iconParams.setMargins(0, 0, dpToPx(12), 0)
            layoutParams = iconParams
        }

        if (model.isUserCreated) {
            // Purple 3x3 grid icon for user-created rooms (matching iOS circle.grid.3x3.fill)
            val gridIcon = ImageView(this).apply {
                setImageResource(R.drawable.ic_grid_3x3)
                val iconSize = dpToPx(28)
                layoutParams = LinearLayout.LayoutParams(iconSize, iconSize)
            }
            iconContainer.addView(gridIcon)
        } else {
            // Green 3D box icon for bundled models
            val boxIcon = TextView(this).apply {
                text = "\uD83D\uDCE6" // Box emoji
                textSize = 20f
                setTextColor(accentGreen)
                gravity = Gravity.CENTER
            }
            iconContainer.addView(boxIcon)
        }
        card.addView(iconContainer)

        // Text container
        val textContainer = LinearLayout(this)
        textContainer.orientation = LinearLayout.VERTICAL
        textContainer.layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)

        // Room name
        val nameText = TextView(this)
        nameText.text = model.name
        nameText.textSize = 17f
        nameText.setTypeface(null, Typeface.BOLD)
        nameText.setTextColor(primaryTextColor)
        textContainer.addView(nameText)

        // Type subtitle
        val typeText = TextView(this)
        typeText.text = getString(R.string.home_3d_room_model)
        typeText.textSize = 13f
        typeText.setTextColor(secondaryTextColor)
        typeText.setPadding(0, dpToPx(2), 0, 0)
        textContainer.addView(typeText)

        // Details row (size and orientation)
        val detailsText = TextView(this)
        val fileSize = getFileSize(model)
        val orientation = if (model.isUserCreated) {
            val orient = if (model.photoOrientation == "landscape") "Landscape" else "Portrait"
            "$orient - held ${if (model.photoOrientation == "landscape") "horizontally" else "vertically"}"
        } else "3D Model"
        // Show actual dimensions if available
        val dimensionStr = if (model.roomWidth != null && model.roomHeight != null) {
            String.format("%.1f × %.1f m", model.roomWidth, model.roomHeight)
        } else {
            "3D Room"
        }
        detailsText.text = "$dimensionStr  •  $fileSize  •  $orientation"
        detailsText.textSize = 13f
        detailsText.setTextColor(secondaryTextColor)
        detailsText.setPadding(0, dpToPx(2), 0, 0)
        textContainer.addView(detailsText)

        card.addView(textContainer)

        // Double chevron
        val chevron = TextView(this)
        chevron.text = "〉〉"
        chevron.textSize = 16f
        chevron.setTextColor(secondaryTextColor)
        chevron.setPadding(dpToPx(8), 0, 0, 0)
        card.addView(chevron)

        // Click listener - use captured model for this card so correct room opens
        val clickedModel = model
        card.setOnClickListener { _ ->
            Toast.makeText(this, getString(R.string.home_opening, clickedModel.name), Toast.LENGTH_SHORT).show()
            LogUtil.d("ContentActivity", "Room clicked: name=${clickedModel.name} id=${clickedModel.id} isUserCreated=${clickedModel.isUserCreated} assetPath=${clickedModel.assetPath}")

            if (clickedModel.isUserCreated) {
                if (clickedModel.assetPath.endsWith(".ply")) {
                    // Open SharpRoomActivity for PLY files (Gaussian splat)
                    val plyFile = File(clickedModel.assetPath)
                    val roomFolder = plyFile.parentFile
                    LogUtil.d("ContentActivity", "Branch PLY: roomFolder=${roomFolder?.absolutePath} starting SharpRoomActivity")
                    val intent = Intent(this, SharpRoomActivity::class.java)
                    intent.putExtra(SharpRoomActivity.EXTRA_PLY_PATH, plyFile.absolutePath)
                    intent.putExtra(SharpRoomActivity.EXTRA_ROOM_FOLDER, roomFolder?.absolutePath)
                    intent.putExtra(SharpRoomActivity.EXTRA_ALLOW_SAVE, false)
                    clickedModel.roomWidth?.let { intent.putExtra(SharpRoomActivity.EXTRA_ROOM_WIDTH, it) }
                    clickedModel.roomHeight?.let { intent.putExtra(SharpRoomActivity.EXTRA_ROOM_HEIGHT, it) }
                    clickedModel.roomDepth?.let { intent.putExtra(SharpRoomActivity.EXTRA_ROOM_DEPTH, it) }
                    clickedModel.roomCenterX?.let { intent.putExtra(SharpRoomActivity.EXTRA_ROOM_CENTER_X, it) }
                    clickedModel.roomCenterY?.let { intent.putExtra(SharpRoomActivity.EXTRA_ROOM_CENTER_Y, it) }
                    clickedModel.roomCenterZ?.let { intent.putExtra(SharpRoomActivity.EXTRA_ROOM_CENTER_Z, it) }
                    intent.putExtra(
                        "photo_orientation",
                        clickedModel.photoOrientation.trim().lowercase().takeIf { it == "landscape" } ?: "portrait"
                    )
                    intent.putExtra(SharpRoomActivity.EXTRA_PHOTO_WIDE_ANGLE, clickedModel.photoWideAngle)
                    LogUtil.d("ContentActivity", "Opening SharpRoomActivity photo_orientation=${clickedModel.photoOrientation} photoWideAngle=${clickedModel.photoWideAngle} roomId=${clickedModel.id}")
                    startActivity(intent)
                } else if (clickedModel.assetPath.endsWith(".glb")) {
                    // Open WebGL-based GLBRoomActivity for GLB files (matching iOS)
                    val glbFile = File(clickedModel.assetPath)
                    val roomFolderPath = glbFile.parent
                    LogUtil.d("ContentActivity", "Branch GLB: path=${clickedModel.assetPath} roomFolder=$roomFolderPath starting GLBRoomActivity")
                    val intent = Intent(this, GLBRoomActivity::class.java)
                    intent.putExtra(GLBRoomActivity.EXTRA_GLB_PATH, clickedModel.assetPath)
                    intent.putExtra(GLBRoomActivity.EXTRA_ROOM_NAME, clickedModel.name)
                    intent.putExtra(GLBRoomActivity.EXTRA_ROOM_ID, clickedModel.id)
                    if (roomFolderPath != null) intent.putExtra("ROOM_FOLDER", roomFolderPath)
                    intent.putExtra(GLBRoomActivity.EXTRA_IS_PREVIEW, false)
                    clickedModel.roomWidth?.let { intent.putExtra(GLBRoomActivity.EXTRA_ROOM_WIDTH, it) }
                    clickedModel.roomHeight?.let { intent.putExtra(GLBRoomActivity.EXTRA_ROOM_HEIGHT, it) }
                    intent.putExtra(GLBRoomActivity.EXTRA_PHOTO_ORIENTATION, clickedModel.photoOrientation)
                    startActivity(intent)
                } else {
                    // Check for files in room folder (assetPath is folder path)
                    val roomFolder = File(clickedModel.assetPath)
                    val plyFile = File(roomFolder, "room.ply")
                    val glbFile = File(roomFolder, "room.glb")
                    LogUtil.d("ContentActivity", "Branch folder: ${roomFolder.absolutePath} plyExists=${plyFile.exists()} glbExists=${glbFile.exists()}")

                    when {
                        plyFile.exists() -> {
                            LogUtil.d("ContentActivity", "Opening SharpRoomActivity with PLY: ${plyFile.absolutePath}, roomFolder=${roomFolder.absolutePath}")
                            val intent = Intent(this, SharpRoomActivity::class.java)
                            intent.putExtra(SharpRoomActivity.EXTRA_PLY_PATH, plyFile.absolutePath)
                            intent.putExtra(SharpRoomActivity.EXTRA_ROOM_FOLDER, roomFolder.absolutePath)
                            intent.putExtra(SharpRoomActivity.EXTRA_ALLOW_SAVE, false)
                            clickedModel.roomWidth?.let { intent.putExtra(SharpRoomActivity.EXTRA_ROOM_WIDTH, it) }
                            clickedModel.roomHeight?.let { intent.putExtra(SharpRoomActivity.EXTRA_ROOM_HEIGHT, it) }
                            clickedModel.roomDepth?.let { intent.putExtra(SharpRoomActivity.EXTRA_ROOM_DEPTH, it) }
                            clickedModel.roomCenterX?.let { intent.putExtra(SharpRoomActivity.EXTRA_ROOM_CENTER_X, it) }
                            clickedModel.roomCenterY?.let { intent.putExtra(SharpRoomActivity.EXTRA_ROOM_CENTER_Y, it) }
                            clickedModel.roomCenterZ?.let { intent.putExtra(SharpRoomActivity.EXTRA_ROOM_CENTER_Z, it) }
                            intent.putExtra(
                                "photo_orientation",
                                clickedModel.photoOrientation.trim().lowercase().takeIf { it == "landscape" } ?: "portrait"
                            )
                            intent.putExtra(SharpRoomActivity.EXTRA_PHOTO_WIDE_ANGLE, clickedModel.photoWideAngle)
                            LogUtil.d("ContentActivity", "Opening SharpRoomActivity (folder) photo_orientation=${clickedModel.photoOrientation} photoWideAngle=${clickedModel.photoWideAngle} roomId=${clickedModel.id}")
                            startActivity(intent)
                        }
                        glbFile.exists() -> {
                            LogUtil.d("ContentActivity", "Opening GLBRoomActivity with GLB: ${glbFile.absolutePath}, roomFolder=${roomFolder.absolutePath}")
                            val intent = Intent(this, GLBRoomActivity::class.java)
                            intent.putExtra(GLBRoomActivity.EXTRA_GLB_PATH, glbFile.absolutePath)
                            intent.putExtra(GLBRoomActivity.EXTRA_ROOM_NAME, clickedModel.name)
                            intent.putExtra(GLBRoomActivity.EXTRA_ROOM_ID, clickedModel.id)
                            intent.putExtra("ROOM_FOLDER", roomFolder.absolutePath)
                            intent.putExtra(GLBRoomActivity.EXTRA_IS_PREVIEW, false)
                            clickedModel.roomWidth?.let { intent.putExtra(GLBRoomActivity.EXTRA_ROOM_WIDTH, it) }
                            clickedModel.roomHeight?.let { intent.putExtra(GLBRoomActivity.EXTRA_ROOM_HEIGHT, it) }
                            intent.putExtra(GLBRoomActivity.EXTRA_PHOTO_ORIENTATION, clickedModel.photoOrientation)
                            startActivity(intent)
                        }
                        else -> {
                            LogUtil.d("ContentActivity", "No PLY/GLB in folder, opening RoomViewerActivity: ${clickedModel.assetPath}")
                            val intent = Intent(this, RoomViewerActivity::class.java)
                            intent.putExtra(RoomViewerActivity.EXTRA_ROOM_FOLDER, clickedModel.assetPath)
                            startActivity(intent)
                        }
                    }
                }
            } else {
                LogUtil.d("ContentActivity", "Branch bundled: opening ModelDetailActivity with id=${clickedModel.id} (name=${clickedModel.name})")
                val intent = Intent(this, ModelDetailActivity::class.java)
                intent.putExtra("MODEL_ID", clickedModel.id)
                startActivity(intent)
            }
        }

        // Long press: rename or delete (user-created rooms only)
        if (model.isUserCreated) {
            val longPressModel = model
            card.setOnLongClickListener {
                showRoomActionsDialog(longPressModel)
                true
            }
        }

        return card
    }

    private fun getFileSize(model: Model): String {
        try {
            val file = File(model.assetPath)
            if (file.exists()) {
                val bytes = if (file.isDirectory) {
                    file.walkTopDown().filter { it.isFile }.map { it.length() }.sum()
                } else {
                    file.length()
                }
                val mb = bytes / (1024.0 * 1024.0)
                return if (mb >= 1) {
                    String.format("%.1f MB", mb)
                } else {
                    String.format("%.0f KB", bytes / 1024.0)
                }
            }
        } catch (e: Exception) {
            // Ignore
        }
        return "—"
    }

    private fun showRoomActionsDialog(model: Model) {
        val options = arrayOf(
            getString(R.string.home_rename_room),
            getString(R.string.common_delete),
        )
        AlertDialog.Builder(this)
            .setTitle(model.name)
            .setItems(options) { _, which ->
                when (which) {
                    0 -> showRenameRoomDialog(model)
                    1 -> showDeleteDialog(model)
                }
            }
            .setNegativeButton(R.string.common_cancel, null)
            .show()
    }

    private fun showRenameRoomDialog(model: Model) {
        val padding = dpToPx(24)
        val input = EditText(this).apply {
            setText(model.name)
            setSelection(model.name.length)
            setHint(R.string.home_rename_room_hint)
        }
        val container = FrameLayout(this).apply {
            setPadding(padding, dpToPx(8), padding, 0)
            addView(
                input,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ),
            )
        }
        val dialog = AlertDialog.Builder(this)
            .setTitle(R.string.home_rename_room_title)
            .setView(container)
            .setPositiveButton(R.string.common_save, null)
            .setNegativeButton(R.string.common_cancel, null)
            .create()
        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val name = input.text?.toString()?.trim().orEmpty()
                if (name.isEmpty()) {
                    Toast.makeText(this, getString(R.string.home_rename_room_empty), Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }
                if (!modelManager.isRoomNameAvailable(name, excludeRoomId = model.id)) {
                    Toast.makeText(this, getString(R.string.home_room_name_duplicate), Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }
                if (modelManager.renameUserRoom(model.id, name)) {
                    Toast.makeText(this, getString(R.string.home_room_renamed, name), Toast.LENGTH_SHORT).show()
                    refreshRoomsList()
                    dialog.dismiss()
                } else {
                    Toast.makeText(this, getString(R.string.home_rename_room_failed), Toast.LENGTH_SHORT).show()
                }
            }
        }
        dialog.show()
    }

    private fun showDeleteDialog(model: Model) {
        AlertDialog.Builder(this)
            .setTitle("Delete Room")
            .setMessage("Are you sure you want to delete \"${model.name}\"?")
            .setPositiveButton("Delete") { _, _ ->
                deleteRoom(model)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun deleteRoom(model: Model) {
        try {
            modelManager.deleteRoom(model.id)
            Toast.makeText(this, getString(R.string.deleted_room, model.name), Toast.LENGTH_SHORT).show()
            refreshRoomsList()
        } catch (e: Exception) {
            Toast.makeText(this, getString(R.string.home_failed_delete, e.message ?: ""), Toast.LENGTH_SHORT).show()
            CrashReporter.report(this, e, "Content — delete room")
        }
    }
}
