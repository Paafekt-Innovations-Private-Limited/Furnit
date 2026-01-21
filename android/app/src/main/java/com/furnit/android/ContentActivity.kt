package com.furnit.android

import android.content.Intent
import android.graphics.BitmapFactory
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
import com.furnit.android.models.Model
import com.furnit.android.models.ModelManager

class ContentActivity : AppCompatActivity() {
    private lateinit var modelManager: ModelManager
    private lateinit var roomsContainer: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        modelManager = ModelManager(this)
        setupUI()
    }

    override fun onResume() {
        super.onResume()
        // Refresh models when returning to this activity
        modelManager.refresh()
        refreshRoomsList()
    }

    private fun setupUI() {
        val scrollView = ScrollView(this).apply {
            setBackgroundColor(Color.parseColor("#F5F5F5"))
        }

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 48, 32, 32)
        }

        // Title
        val title = TextView(this).apply {
            text = "Furnit"
            textSize = 28f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#333333"))
            setPadding(0, 0, 0, 8)
        }
        layout.addView(title)

        // Subtitle
        val subtitle = TextView(this).apply {
            text = "Your Rooms"
            textSize = 16f
            setTextColor(Color.parseColor("#666666"))
            setPadding(0, 0, 0, 16)
        }
        layout.addView(subtitle)

        // Create Room button
        val createRoomBtn = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(Color.parseColor("#4CAF50"))
            setPadding(24, 16, 24, 16)
            gravity = Gravity.CENTER
            val params = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
            params.setMargins(0, 0, 0, 24)
            layoutParams = params

            val icon = TextView(this@ContentActivity).apply {
                text = "+"
                textSize = 20f
                setTextColor(Color.WHITE)
                setTypeface(null, Typeface.BOLD)
                setPadding(0, 0, 12, 0)
            }
            addView(icon)

            val btnText = TextView(this@ContentActivity).apply {
                text = "Create Room"
                textSize = 16f
                setTextColor(Color.WHITE)
                setTypeface(null, Typeface.BOLD)
            }
            addView(btnText)

            setOnClickListener {
                startActivity(Intent(this@ContentActivity, SinglePhotoRoomActivity::class.java))
            }
        }
        layout.addView(createRoomBtn)

        // Rooms container (will be refreshed)
        roomsContainer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        layout.addView(roomsContainer)

        // Settings button at bottom
        val settingsBtn = TextView(this).apply {
            text = "Settings"
            textSize = 16f
            setTextColor(Color.parseColor("#666666"))
            gravity = Gravity.CENTER
            setPadding(0, 48, 0, 0)
            setOnClickListener {
                startActivity(Intent(this@ContentActivity, SettingsActivity::class.java))
            }
        }
        layout.addView(settingsBtn)

        scrollView.addView(layout)
        setContentView(scrollView)

        // Initial load
        refreshRoomsList()
    }

    private fun refreshRoomsList() {
        roomsContainer.removeAllViews()

        val models = modelManager.listModels()
        Log.d("ContentActivity", "Loaded ${models.size} models:")
        models.forEach { model ->
            Log.d("ContentActivity", "  - ${model.name} (id=${model.id}, isUserCreated=${model.isUserCreated}, path=${model.assetPath})")
        }

        if (models.isEmpty()) {
            // Empty state
            val emptyText = TextView(this).apply {
                text = "No rooms yet\nTap 'Create Room' to get started"
                textSize = 16f
                setTextColor(Color.parseColor("#999999"))
                gravity = Gravity.CENTER
                setPadding(0, 60, 0, 60)
            }
            roomsContainer.addView(emptyText)
        } else {
            // Room cards
            for (model in models) {
                val card = createRoomCard(model)
                roomsContainer.addView(card)
            }
        }
    }

    private fun createRoomCard(model: Model): View {
        val card = LinearLayout(this)
        card.orientation = LinearLayout.HORIZONTAL
        card.setBackgroundColor(Color.WHITE)
        card.setPadding(16, 16, 16, 16)
        card.gravity = Gravity.CENTER_VERTICAL

        val params = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )
        params.setMargins(0, 0, 0, 12)
        card.layoutParams = params

        // Thumbnail for user-created rooms
        if (model.isUserCreated && model.thumbnailPath != null) {
            val thumbnail = ImageView(this)
            try {
                val bitmap = BitmapFactory.decodeFile(model.thumbnailPath)
                thumbnail.setImageBitmap(bitmap)
            } catch (e: Exception) {
                thumbnail.setBackgroundColor(Color.parseColor("#E0E0E0"))
            }
            thumbnail.scaleType = ImageView.ScaleType.CENTER_CROP
            val thumbParams = LinearLayout.LayoutParams(80, 80)
            thumbParams.setMargins(0, 0, 16, 0)
            thumbnail.layoutParams = thumbParams
            card.addView(thumbnail)
        } else {
            // Placeholder for bundled models
            val placeholder = View(this)
            placeholder.setBackgroundColor(Color.parseColor("#E8F5E9"))
            val placeholderParams = LinearLayout.LayoutParams(80, 80)
            placeholderParams.setMargins(0, 0, 16, 0)
            placeholder.layoutParams = placeholderParams
            card.addView(placeholder)
        }

        // Text container
        val textContainer = LinearLayout(this)
        textContainer.orientation = LinearLayout.VERTICAL
        textContainer.layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)

        // Room name
        val nameText = TextView(this)
        nameText.text = model.name
        nameText.textSize = 16f
        nameText.setTypeface(null, Typeface.BOLD)
        nameText.setTextColor(Color.parseColor("#333333"))
        textContainer.addView(nameText)

        // Type hint
        val hintText = TextView(this)
        hintText.text = if (model.isUserCreated) "Photo Room" else "3D Model"
        hintText.textSize = 12f
        hintText.setTextColor(Color.parseColor("#999999"))
        hintText.setPadding(0, 4, 0, 0)
        textContainer.addView(hintText)

        card.addView(textContainer)

        // Chevron
        val chevron = TextView(this)
        chevron.text = ">"
        chevron.textSize = 18f
        chevron.setTextColor(Color.parseColor("#CCCCCC"))
        card.addView(chevron)

        // Click listener - set after all views are added
        card.setOnClickListener { v ->
            Toast.makeText(this, "Opening ${model.name}...", Toast.LENGTH_SHORT).show()
            Log.d("ContentActivity", "Room clicked: ${model.name}, isUserCreated=${model.isUserCreated}, assetPath=${model.assetPath}")

            if (model.isUserCreated) {
                Log.d("ContentActivity", "Opening RoomViewerActivity with folder: ${model.assetPath}")
                val intent = Intent(this, RoomViewerActivity::class.java)
                intent.putExtra(RoomViewerActivity.EXTRA_ROOM_FOLDER, model.assetPath)
                startActivity(intent)
            } else {
                Log.d("ContentActivity", "Opening ModelDetailActivity with id: ${model.id}")
                val intent = Intent(this, ModelDetailActivity::class.java)
                intent.putExtra("MODEL_ID", model.id)
                startActivity(intent)
            }
        }

        return card
    }
}
