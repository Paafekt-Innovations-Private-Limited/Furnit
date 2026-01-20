package com.furnit.android

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import android.widget.LinearLayout
import android.view.Gravity
import android.widget.TextView

class LoginActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Minimal login UI: a button to continue to the main content
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(40, 40, 40, 40)
        }

        val title = TextView(this).apply {
            text = "Furnit - Login"
            textSize = 22f
            gravity = Gravity.CENTER
        }

        val btn = MaterialButton(this).apply {
            text = "Continue"
            setOnClickListener {
                startActivity(Intent(this@LoginActivity, ContentActivity::class.java))
                finish()
            }
        }

        layout.addView(title)
        layout.addView(btn)

        setContentView(layout)
    }
}
