package com.furnit.android

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import android.widget.LinearLayout
import android.view.Gravity
import android.widget.TextView
import android.view.ViewGroup
import android.graphics.Color
import android.graphics.Typeface
import android.view.View
import com.furnit.android.models.ModelManager

class ContentActivity : AppCompatActivity() {
    private lateinit var modelManager: ModelManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        modelManager = ModelManager(this)

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 48, 32, 32)
            setBackgroundColor(Color.parseColor("#F5F5F5"))
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
            setPadding(0, 0, 0, 32)
        }
        layout.addView(subtitle)

        val models = modelManager.listModels()

        if (models.isEmpty()) {
            // Empty state
            val emptyText = TextView(this).apply {
                text = "No rooms yet"
                textSize = 18f
                setTextColor(Color.parseColor("#999999"))
                gravity = Gravity.CENTER
                setPadding(0, 100, 0, 0)
            }
            layout.addView(emptyText)
        } else {
            // Room cards
            for (model in models) {
                val card = createRoomCard(model.id, model.name)
                layout.addView(card)
            }
        }

        // Settings button at bottom
        val settingsBtn = TextView(this).apply {
            text = "⚙️ Settings"
            textSize = 16f
            setTextColor(Color.parseColor("#666666"))
            gravity = Gravity.CENTER
            setPadding(0, 48, 0, 0)
            setOnClickListener {
                startActivity(Intent(this@ContentActivity, SettingsActivity::class.java))
            }
        }
        layout.addView(settingsBtn)

        setContentView(layout)
    }

    private fun createRoomCard(modelId: String, modelName: String): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.WHITE)
            setPadding(24, 24, 24, 24)
            val params = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
            params.setMargins(0, 0, 0, 16)
            layoutParams = params

            // Room name
            val nameText = TextView(this@ContentActivity).apply {
                text = modelName
                textSize = 18f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.parseColor("#333333"))
            }
            addView(nameText)

            // Tap hint
            val hintText = TextView(this@ContentActivity).apply {
                text = "Tap to view"
                textSize = 14f
                setTextColor(Color.parseColor("#999999"))
                setPadding(0, 8, 0, 0)
            }
            addView(hintText)

            // Click to open room
            setOnClickListener {
                val intent = Intent(this@ContentActivity, ModelDetailActivity::class.java)
                intent.putExtra("MODEL_ID", modelId)
                startActivity(intent)
            }
        }
    }
}
