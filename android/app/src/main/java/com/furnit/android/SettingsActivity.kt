package com.furnit.android

import android.content.SharedPreferences
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import android.widget.LinearLayout
import android.view.Gravity
import android.widget.RadioGroup
import android.widget.RadioButton
import android.widget.TextView
import android.content.Context
import android.view.View
import com.furnit.android.models.QualitySettings

class SettingsActivity : AppCompatActivity() {
    private lateinit var prefs: SharedPreferences

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = getSharedPreferences("furnit_prefs", Context.MODE_PRIVATE)

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(20,20,20,20)
        }

        val title = TextView(this).apply { text = "Settings"; textSize = 20f }

        val rg = RadioGroup(this)
        val lowId = View.generateViewId()
        val medId = View.generateViewId()
        val highId = View.generateViewId()
        val low = RadioButton(this).apply { text = "Low"; id = lowId }
        val med = RadioButton(this).apply { text = "Medium"; id = medId }
        val high = RadioButton(this).apply { text = "High"; id = highId }
        rg.addView(low); rg.addView(med); rg.addView(high)

        val current = prefs.getString("quality", QualitySettings.MEDIUM)
        when (current) {
            QualitySettings.LOW -> rg.check(lowId)
            QualitySettings.MEDIUM -> rg.check(medId)
            QualitySettings.HIGH -> rg.check(highId)
        }

        rg.setOnCheckedChangeListener { _, checkedId ->
            val v = when (checkedId) {
                lowId -> QualitySettings.LOW
                medId -> QualitySettings.MEDIUM
                highId -> QualitySettings.HIGH
                else -> QualitySettings.MEDIUM
            }
            prefs.edit().putString("quality", v).apply()
        }

        layout.addView(title)
        layout.addView(rg)
        setContentView(layout)
    }
}
