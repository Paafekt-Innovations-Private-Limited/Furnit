package com.furnit.android

import android.content.Intent
import android.graphics.BitmapFactory
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import android.widget.LinearLayout
import android.view.Gravity
import android.widget.TextView
import android.widget.ImageView
import android.view.ViewGroup
import android.graphics.Color
import android.graphics.Typeface
import android.view.View
import android.widget.ScrollView
import android.widget.Toast
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

        // Create Room icon button (left)
        val createRoomIcon = createIconButton("\uD83D\uDDBC") // Image icon
        createRoomIcon.setOnClickListener {
            startActivity(Intent(this@ContentActivity, SinglePhotoRoomActivity::class.java))
        }
        topBar.addView(createRoomIcon)

        // Spacer
        val spacer = View(this).apply {
            layoutParams = LinearLayout.LayoutParams(0, 1, 1f)
        }
        topBar.addView(spacer)

        // Help icon button
        val helpIcon = createIconButton("?")
        helpIcon.setOnClickListener {
            showHelpDialog()
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
            text = "0 of 1000 rooms remaining"
            textSize = 14f
            setTextColor(primaryTextColor)
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        statsRow.addView(statsText)

        val totalLabel = TextView(this).apply {
            text = "Total"
            textSize = 12f
            setTextColor(secondaryTextColor)
            setPadding(0, 0, dpToPx(8), 0)
        }
        statsRow.addView(totalLabel)

        totalSizeText = TextView(this).apply {
            text = "0 MB"
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
            text = "Long press to delete"
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

        // Initial load
        refreshRoomsList()
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

    private fun showHelpDialog() {
        AlertDialog.Builder(this)
            .setTitle("Help")
            .setMessage("• Tap the image icon to create a new room from a photo\n\n• Long press on a room to delete it\n\n• Tap a room to view it in 3D")
            .setPositiveButton("OK", null)
            .show()
    }

    private fun refreshRoomsList() {
        roomsContainer.removeAllViews()

        val models = modelManager.listModels()
        Log.d("ContentActivity", "Loaded ${models.size} models:")
        models.forEach { model ->
            Log.d("ContentActivity", "  - ${model.name} (id=${model.id}, isUserCreated=${model.isUserCreated}, path=${model.assetPath})")
        }

        // Update stats
        val roomCount = models.size
        val remaining = 1000 - roomCount
        statsText.text = "$remaining of 1000 rooms remaining"

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
                text = "No rooms yet\nTap the image icon to get started"
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
            // Purple grid icon for user-created rooms
            val gridIcon = TextView(this).apply {
                text = "\u25A6\u25A6\u25A6\n\u25A6\u25A6\u25A6\n\u25A6\u25A6\u25A6"
                textSize = 8f
                setTextColor(accentPurple)
                gravity = Gravity.CENTER
                setLineSpacing(0f, 0.8f)
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
        typeText.text = "3D Room Model"
        typeText.textSize = 13f
        typeText.setTextColor(secondaryTextColor)
        typeText.setPadding(0, dpToPx(2), 0, 0)
        textContainer.addView(typeText)

        // Details row (size and orientation)
        val detailsText = TextView(this)
        val fileSize = getFileSize(model)
        val orientation = if (model.isUserCreated) "Portrait - held vertically" else "3D Model"
        detailsText.text = "3D Room  •  $fileSize  •  $orientation"
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

        // Click listener
        card.setOnClickListener { v ->
            Toast.makeText(this, "Opening ${model.name}...", Toast.LENGTH_SHORT).show()
            Log.d("ContentActivity", "Room clicked: ${model.name}, isUserCreated=${model.isUserCreated}, assetPath=${model.assetPath}")

            if (model.isUserCreated) {
                if (model.assetPath.endsWith(".glb")) {
                    Log.d("ContentActivity", "Opening ModelDetailActivity with GLB: ${model.assetPath}")
                    val intent = Intent(this, ModelDetailActivity::class.java)
                    intent.putExtra("MODEL_ID", model.id)
                    startActivity(intent)
                } else {
                    Log.d("ContentActivity", "Opening RoomViewerActivity with folder: ${model.assetPath}")
                    val intent = Intent(this, RoomViewerActivity::class.java)
                    intent.putExtra(RoomViewerActivity.EXTRA_ROOM_FOLDER, model.assetPath)
                    startActivity(intent)
                }
            } else {
                Log.d("ContentActivity", "Opening ModelDetailActivity with id: ${model.id}")
                val intent = Intent(this, ModelDetailActivity::class.java)
                intent.putExtra("MODEL_ID", model.id)
                startActivity(intent)
            }
        }

        // Long press to delete
        card.setOnLongClickListener { v ->
            showDeleteDialog(model)
            true
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
            Toast.makeText(this, "Deleted ${model.name}", Toast.LENGTH_SHORT).show()
            refreshRoomsList()
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to delete: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
}
