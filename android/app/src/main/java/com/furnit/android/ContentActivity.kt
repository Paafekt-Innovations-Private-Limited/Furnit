package com.furnit.android

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import android.widget.LinearLayout
import android.view.Gravity
import android.widget.Button
import android.widget.TextView

class ContentActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(40,40,40,40)
        }

        val title = TextView(this).apply {
            text = "Furnit - Content"
            textSize = 20f
        }

        val arButton = Button(this).apply {
            text = "Open AR"
            setOnClickListener {
                startActivity(Intent(this@ContentActivity, ARActivity::class.java))
            }
        }

        val viewerButton = Button(this).apply {
            text = "Open Model Viewer"
            setOnClickListener {
                startActivity(Intent(this@ContentActivity, ModelViewerActivity::class.java))
            }
        }

        val settingsButton = Button(this).apply {
            text = "Settings"
            setOnClickListener {
                startActivity(Intent(this@ContentActivity, SettingsActivity::class.java))
            }
        }

        layout.addView(title)
        layout.addView(arButton)
        layout.addView(viewerButton)
        layout.addView(settingsButton)

        setContentView(layout)
    }
}
