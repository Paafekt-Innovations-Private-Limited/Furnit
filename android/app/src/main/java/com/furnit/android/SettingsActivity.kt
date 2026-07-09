package com.furnit.android

import android.content.Intent
import android.content.SharedPreferences
import android.net.Uri
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.*
import android.content.res.ColorStateList
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import com.furnit.android.auth.AuthenticationManager
import com.furnit.android.auth.LoginActivity
import com.furnit.android.utils.DebugLogger

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
            text = "< ${getString(R.string.common_back)}"
            textSize = 16f
            setTextColor(Color.parseColor("#007AFF"))
            setPadding(0, 0, 0, 24)
            setOnClickListener { finish() }
        }
        layout.addView(backBtn)

        // Title
        val title = TextView(this).apply {
            text = getString(R.string.settings_title)
            textSize = 24f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#333333"))
            setPadding(0, 0, 0, 32)
        }
        layout.addView(title)

        // User info section
        val userSection = createSection(getString(R.string.settings_account))
        val userInfo = TextView(this).apply {
            val userName = authManager.getUserName()
            val userPhone = authManager.getUserPhone()
            text = if (userName.isNotEmpty()) "$userName\n$userPhone" else userPhone.ifEmpty { getString(R.string.settings_not_signed_in) }
            textSize = 16f
            setTextColor(Color.parseColor("#333333"))
            setPadding(0, 8, 0, 16)
        }
        userSection.addView(userInfo)
        layout.addView(userSection)

        val furnitureFitSection = createSection(getString(R.string.settings_furniture_segmentation))

        val imageScanRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 12, 0, 8)
            isClickable = true
            isFocusable = true
            setOnClickListener {
                startActivity(Intent(this@SettingsActivity, SettingsImageScanActivity::class.java))
            }
        }
        val imageScanLabel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        imageScanLabel.addView(
            TextView(this).apply {
                text = getString(R.string.settings_image_scan)
                textSize = 16f
                setTextColor(Color.parseColor("#333333"))
            },
        )
        imageScanLabel.addView(
            TextView(this).apply {
                text = getString(R.string.settings_image_scan_description)
                textSize = 12f
                setTextColor(Color.parseColor("#666666"))
            },
        )
        val imageScanChevron = TextView(this).apply {
            text = ">"
            textSize = 18f
            setTextColor(Color.parseColor("#999999"))
            setPadding(24, 0, 0, 0)
        }
        imageScanRow.addView(imageScanLabel)
        imageScanRow.addView(imageScanChevron)
        furnitureFitSection.addView(imageScanRow)

        layout.addView(furnitureFitSection)

        // Room Viewer settings section
        val viewerSection = createSection(getString(R.string.settings_room_viewer))

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
            text = getString(R.string.settings_auto_orbit)
            textSize = 16f
            setTextColor(Color.parseColor("#333333"))
        }
        val autoOrbitDesc = TextView(this).apply {
            text = getString(R.string.settings_auto_orbit_description)
            textSize = 12f
            setTextColor(Color.parseColor("#666666"))
        }
        autoOrbitLabel.addView(autoOrbitTitle)
        autoOrbitLabel.addView(autoOrbitDesc)

        val autoOrbitSwitch = createStyledSwitch(prefs.getBoolean("auto_orbit_enabled", false)) { isChecked ->
            prefs.edit().putBoolean("auto_orbit_enabled", isChecked).apply()
        }

        autoOrbitLayout.addView(autoOrbitLabel)
        autoOrbitLayout.addView(autoOrbitSwitch)
        viewerSection.addView(autoOrbitLayout)
        layout.addView(viewerSection)

        val roomDefaultsSection = createSection(getString(R.string.settings_room_dimensions))
        roomDefaultsSection.addView(
            TextView(this).apply {
                text = getString(R.string.settings_room_dimensions_footer)
                textSize = 12f
                setTextColor(Color.parseColor("#666666"))
                setPadding(0, 4, 0, 12)
            },
        )
        roomDefaultsSection.addView(
            createDimensionSliderRow(
                label = getString(R.string.settings_width_label),
                initialValue = RoomDefaults.widthMeters(prefs),
                min = 2.0f,
                max = 10.0f,
                onValueChanged = { RoomDefaults.setWidthMeters(prefs, it) },
            ),
        )
        roomDefaultsSection.addView(
            createDimensionSliderRow(
                label = getString(R.string.settings_height_label),
                initialValue = RoomDefaults.heightMeters(prefs),
                min = 2.0f,
                max = 5.0f,
                onValueChanged = { RoomDefaults.setHeightMeters(prefs, it) },
            ),
        )
        roomDefaultsSection.addView(
            createDimensionSliderRow(
                label = getString(R.string.settings_depth_label),
                initialValue = RoomDefaults.depthMeters(prefs),
                min = 2.0f,
                max = 10.0f,
                onValueChanged = { RoomDefaults.setDepthMeters(prefs, it) },
            ),
        )
        layout.addView(roomDefaultsSection)

        // Developer Settings section (matches iOS)
        val developerSection = createSection(getString(R.string.settings_developer))

        // Debug mode toggle
        val debugLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 8, 0, 8)
        }

        val debugLabel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }

        val debugIcon = TextView(this).apply {
            text = "\uD83D\uDC1E" // Bug/ladybug emoji
            textSize = 20f
            setPadding(0, 0, 12, 0)
        }

        val debugTextContainer = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        debugTextContainer.addView(debugIcon)

        val debugTextLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }

        val debugTitle = TextView(this).apply {
            text = getString(R.string.settings_debug_mode)
            textSize = 16f
            setTextColor(Color.parseColor("#333333"))
        }
        val debugDesc = TextView(this).apply {
            text = getString(R.string.settings_debug_mode_description)
            textSize = 12f
            setTextColor(Color.parseColor("#666666"))
        }
        debugTextLayout.addView(debugTitle)
        debugTextLayout.addView(debugDesc)
        debugTextContainer.addView(debugTextLayout)
        debugLabel.addView(debugTextContainer)

        // Initialize DebugLogger
        DebugLogger.init(this)

        val debugSwitch = createStyledSwitch(DebugLogger.isDebugMode) { isChecked ->
            DebugLogger.setDebugMode(isChecked)
        }

        debugLayout.addView(debugLabel)
        debugLayout.addView(debugSwitch)
        developerSection.addView(debugLayout)

        // Developer section footer
        val developerFooter = TextView(this).apply {
            text = getString(R.string.settings_developer_footer)
            textSize = 11f
            setTextColor(Color.parseColor("#999999"))
            setPadding(0, 8, 0, 0)
        }
        developerSection.addView(developerFooter)

        layout.addView(developerSection)

        // Legal section
        val legalSection = createSection(getString(R.string.settings_legal))

        val privacyButton = TextView(this).apply {
            text = getString(R.string.settings_privacy_policy)
            textSize = 16f
            setTextColor(Color.parseColor("#007AFF"))
            setPadding(0, 8, 0, 8)
            setOnClickListener {
                openUrl("https://paafekt.com/privacy")
            }
        }
        legalSection.addView(privacyButton)

        val termsButton = TextView(this).apply {
            text = getString(R.string.settings_terms_of_service)
            textSize = 16f
            setTextColor(Color.parseColor("#007AFF"))
            setPadding(0, 8, 0, 8)
            setOnClickListener {
                openUrl("https://paafekt.com/terms")
            }
        }
        legalSection.addView(termsButton)

        val creditsButton = TextView(this).apply {
            text = getString(R.string.settings_credits)
            textSize = 16f
            setTextColor(Color.parseColor("#007AFF"))
            setPadding(0, 8, 0, 8)
            setOnClickListener {
                startActivity(Intent(this@SettingsActivity, CreditsActivity::class.java))
            }
        }
        legalSection.addView(creditsButton)

        val licenseButton = TextView(this).apply {
            text = getString(R.string.settings_licenses)
            textSize = 16f
            setTextColor(Color.parseColor("#007AFF"))
            setPadding(0, 8, 0, 8)
            setOnClickListener {
                startActivity(Intent(this@SettingsActivity, LicensesActivity::class.java))
            }
        }
        legalSection.addView(licenseButton)

        layout.addView(legalSection)

        // App info section
        val appSection = createSection(getString(R.string.profile_about))
        val versionText = TextView(this).apply {
            text = getString(R.string.app_version_display)
            textSize = 14f
            setTextColor(Color.parseColor("#666666"))
        }
        appSection.addView(versionText)
        layout.addView(appSection)

        // Logout button
        val logoutBtn = Button(this).apply {
            text = getString(R.string.settings_sign_out)
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
            .setTitle(getString(R.string.settings_sign_out))
            .setMessage(getString(R.string.settings_sign_out_confirm))
            .setPositiveButton(getString(R.string.settings_sign_out)) { _, _ ->
                authManager.logout()
                navigateToLogin()
            }
            .setNegativeButton(getString(R.string.common_cancel), null)
            .show()
    }

    private fun navigateToLogin() {
        val intent = Intent(this, LoginActivity::class.java)
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        startActivity(intent)
        finish()
    }

    private fun openUrl(url: String) {
        try {
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            startActivity(intent)
        } catch (e: Exception) {
            // Ignore if no browser available
        }
    }

    private fun createStyledSwitch(checked: Boolean, onChanged: (Boolean) -> Unit): SwitchCompat {
        return SwitchCompat(this).apply {
            isChecked = checked
            // Track colors: gray when off, green when on
            trackTintList = ColorStateList(
                arrayOf(
                    intArrayOf(-android.R.attr.state_checked),
                    intArrayOf(android.R.attr.state_checked)
                ),
                intArrayOf(
                    Color.parseColor("#CCCCCC"),  // Off - gray border
                    Color.parseColor("#81C784")   // On - light green
                )
            )
            // Thumb colors: white when off, green when on
            thumbTintList = ColorStateList(
                arrayOf(
                    intArrayOf(-android.R.attr.state_checked),
                    intArrayOf(android.R.attr.state_checked)
                ),
                intArrayOf(
                    Color.parseColor("#FFFFFF"),  // Off - white
                    Color.parseColor("#4CAF50")   // On - green
                )
            )
            setOnCheckedChangeListener { _, isChecked -> onChanged(isChecked) }
        }
    }

    private fun createDimensionSliderRow(
        label: String,
        initialValue: Float,
        min: Float,
        max: Float,
        onValueChanged: (Float) -> Unit,
    ): LinearLayout {
        val step = 0.1f
        val maxSteps = ((max - min) / step).toInt()

        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, 8, 0, 8)

            val titleRow = LinearLayout(this@SettingsActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
            }

            val labelView = TextView(this@SettingsActivity).apply {
                text = label
                textSize = 16f
                setTextColor(Color.parseColor("#333333"))
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            }

            val valueView = TextView(this@SettingsActivity).apply {
                text = getString(R.string.settings_dimension_value_meters, initialValue)
                textSize = 14f
                setTextColor(Color.parseColor("#666666"))
            }

            titleRow.addView(labelView)
            titleRow.addView(valueView)
            addView(titleRow)

            val seekBar = SeekBar(this@SettingsActivity).apply {
                this.max = maxSteps
                progress = (((initialValue - min) / step).toInt()).coerceIn(0, maxSteps)
                setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                    override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                        val value = (min + progress * step).coerceIn(min, max)
                        valueView.text = getString(R.string.settings_dimension_value_meters, value)
                        onValueChanged(value)
                    }

                    override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit

                    override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
                })
            }
            addView(seekBar)
        }
    }

}
