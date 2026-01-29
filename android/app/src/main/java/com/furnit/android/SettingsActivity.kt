package com.furnit.android

import android.content.Intent
import android.content.SharedPreferences
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.furnit.android.auth.AuthenticationManager
import com.furnit.android.auth.LoginActivity
import com.furnit.android.models.QualitySettings

class SettingsActivity : AppCompatActivity() {
    private lateinit var prefs: SharedPreferences
    private lateinit var authManager: AuthenticationManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = getSharedPreferences("furnit_prefs", MODE_PRIVATE)
        authManager = AuthenticationManager.getInstance(this)

        val scrollView = ScrollView(this).apply {
            setBackgroundColor(Color.parseColor("#F5F5F5"))
        }

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 48, 32, 32)
        }

        // Back button
        val backBtn = TextView(this).apply {
            text = "< Back"
            textSize = 16f
            setTextColor(Color.parseColor("#007AFF"))
            setPadding(0, 0, 0, 24)
            setOnClickListener { finish() }
        }
        layout.addView(backBtn)

        // Title
        val title = TextView(this).apply {
            text = "Settings"
            textSize = 24f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#333333"))
            setPadding(0, 0, 0, 32)
        }
        layout.addView(title)

        // User info section
        val userSection = createSection("Account")
        val userInfo = TextView(this).apply {
            val userName = authManager.getUserName()
            val userPhone = authManager.getUserPhone()
            text = if (userName.isNotEmpty()) "$userName\n$userPhone" else userPhone.ifEmpty { "Not signed in" }
            textSize = 16f
            setTextColor(Color.parseColor("#333333"))
            setPadding(0, 8, 0, 16)
        }
        userSection.addView(userInfo)
        layout.addView(userSection)

        // Quality settings section
        val qualitySection = createSection("Rendering Quality")

        val rg = RadioGroup(this).apply {
            orientation = RadioGroup.VERTICAL
        }
        val lowId = View.generateViewId()
        val medId = View.generateViewId()
        val highId = View.generateViewId()

        val low = RadioButton(this).apply {
            text = "Low - Faster performance"
            id = lowId
            setTextColor(Color.parseColor("#333333"))
        }
        val med = RadioButton(this).apply {
            text = "Medium - Balanced"
            id = medId
            setTextColor(Color.parseColor("#333333"))
        }
        val high = RadioButton(this).apply {
            text = "High - Best quality"
            id = highId
            setTextColor(Color.parseColor("#333333"))
        }
        rg.addView(low)
        rg.addView(med)
        rg.addView(high)

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

        qualitySection.addView(rg)
        layout.addView(qualitySection)

        // Room Viewer settings section
        val viewerSection = createSection("Room Viewer")

        // Auto-orbit toggle (default OFF)
        val autoOrbitLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 8, 0, 8)
        }

        val autoOrbitLabel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        val autoOrbitTitle = TextView(this).apply {
            text = "Auto Orbit"
            textSize = 16f
            setTextColor(Color.parseColor("#333333"))
        }
        val autoOrbitDesc = TextView(this).apply {
            text = "Slowly rotate camera when idle"
            textSize = 12f
            setTextColor(Color.parseColor("#666666"))
        }
        autoOrbitLabel.addView(autoOrbitTitle)
        autoOrbitLabel.addView(autoOrbitDesc)

        val autoOrbitSwitch = Switch(this).apply {
            isChecked = prefs.getBoolean("auto_orbit_enabled", false)  // Default OFF
            setOnCheckedChangeListener { _, isChecked ->
                prefs.edit().putBoolean("auto_orbit_enabled", isChecked).apply()
            }
        }

        autoOrbitLayout.addView(autoOrbitLabel)
        autoOrbitLayout.addView(autoOrbitSwitch)
        viewerSection.addView(autoOrbitLayout)
        layout.addView(viewerSection)

        // App info section
        val appSection = createSection("About")
        val versionText = TextView(this).apply {
            text = "Furnit v1.0.0"
            textSize = 14f
            setTextColor(Color.parseColor("#666666"))
        }
        appSection.addView(versionText)
        layout.addView(appSection)

        // Logout button
        val logoutBtn = Button(this).apply {
            text = "Sign Out"
            textSize = 16f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#F44336"))
            setPadding(24, 16, 24, 16)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 32, 0, 0) }

            setOnClickListener {
                showLogoutConfirmation()
            }
        }
        layout.addView(logoutBtn)

        scrollView.addView(layout)
        setContentView(scrollView)
    }

    private fun createSection(title: String): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.WHITE)
            setPadding(24, 16, 24, 16)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 0, 0, 16) }

            val titleView = TextView(this@SettingsActivity).apply {
                text = title
                textSize = 12f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.parseColor("#999999"))
                setPadding(0, 0, 0, 8)
            }
            addView(titleView)
        }
    }

    private fun showLogoutConfirmation() {
        AlertDialog.Builder(this)
            .setTitle("Sign Out")
            .setMessage("Are you sure you want to sign out?")
            .setPositiveButton("Sign Out") { _, _ ->
                authManager.logout()
                navigateToLogin()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun navigateToLogin() {
        val intent = Intent(this, LoginActivity::class.java)
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        startActivity(intent)
        finish()
    }
}
