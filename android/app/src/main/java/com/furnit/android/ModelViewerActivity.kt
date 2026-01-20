package com.furnit.android

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import android.widget.LinearLayout
import android.view.Gravity
import android.widget.TextView
import com.furnit.android.models.ModelManager
import android.widget.Button
import android.view.ViewGroup

class ModelViewerActivity : AppCompatActivity() {
    private lateinit var modelManager: ModelManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        modelManager = ModelManager(this)

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(20,20,20,20)
        }

        val title = TextView(this).apply { text = "Models"; textSize = 20f }
        layout.addView(title)

        for (model in modelManager.listModels()) {
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            }

            val name = TextView(this).apply {
                text = model.name
                textSize = 16f
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            }

            val btn = Button(this).apply {
                text = "View in AR"
                setOnClickListener {
                    val i = Intent(this@ModelViewerActivity, ARActivity::class.java)
                    i.putExtra("MODEL_ID", model.id)
                    startActivity(i)
                }
            }

            row.addView(name)
            row.addView(btn)
            layout.addView(row)
        }

        setContentView(layout)
    }
}
