package com.furnit.android

import android.content.Intent
import android.content.SharedPreferences
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.furnit.android.auth.AuthenticationManager
import com.furnit.android.auth.LoginActivity
import com.furnit.android.theme.PaafektColors
import com.furnit.android.theme.PaafektScreenViews
import com.furnit.android.theme.PaafektSpace
import com.furnit.android.utils.DebugLogger

class SettingsActivity : AppCompatActivity() {
    private lateinit var prefs: SharedPreferences
    private lateinit var authManager: AuthenticationManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = getSharedPreferences("furnit_prefs", MODE_PRIVATE)
        authManager = AuthenticationManager.getInstance(this)

        val layout = PaafektScreenViews.createScreenColumn(this)

        layout.addView(
            PaafektScreenViews.createBackButton(this, PaafektScreenViews.backLabel(this)) { finish() },
        )
        layout.addView(PaafektScreenViews.createScreenTitle(this, getString(R.string.settings_title)))

        // Account
        layout.addView(createAccountSection())

        // Furniture segmentation
        layout.addView(createFurnitureSection())

        // Room viewer
        layout.addView(createViewerSection())

        // Room dimensions
        layout.addView(createRoomDefaultsSection())

        // Developer
        if (BuildConfig.DEBUG) {
            layout.addView(createDeveloperSection())
        }

        // Legal
        layout.addView(createLegalSection())

        // About
        layout.addView(createAboutSection())

        // Sign out
        layout.addView(
            PaafektScreenViews.createDangerButton(this, getString(R.string.settings_sign_out)) {
                showLogoutConfirmation()
            },
        )
        layout.addView(
            PaafektScreenViews.createDangerButton(this, getString(R.string.settings_delete_account)) {
                showDeleteAccountConfirmation()
            },
        )

        PaafektScreenViews.createScreenScrollView(this).apply {
            addView(layout)
            setContentView(this)
        }
    }

    private fun createAccountSection(): LinearLayout {
        return PaafektScreenViews.createSectionCard(this).apply {
            addView(PaafektScreenViews.createSectionLabel(this@SettingsActivity, getString(R.string.settings_account)))
            val userName = authManager.getUserName()
            val userPhone = authManager.getUserPhone()
            val display = if (userName.isNotEmpty()) {
                "$userName\n$userPhone"
            } else {
                userPhone.ifEmpty { getString(R.string.settings_not_signed_in) }
            }
            addView(PaafektScreenViews.createPrimaryLabel(this@SettingsActivity, display))
            addView(
                PaafektScreenViews.createCaptionLabel(
                    this@SettingsActivity,
                    getString(R.string.settings_account_delete_footer),
                ),
            )
        }
    }

    private fun createFurnitureSection(): LinearLayout {
        return PaafektScreenViews.createSectionCard(this).apply {
            addView(PaafektScreenViews.createSectionLabel(this@SettingsActivity, getString(R.string.settings_furniture_segmentation)))
            addView(
                PaafektScreenViews.createNavRow(
                    this@SettingsActivity,
                    getString(R.string.settings_image_scan),
                    getString(R.string.settings_image_scan_description),
                ) {
                    startActivity(Intent(this@SettingsActivity, SettingsImageScanActivity::class.java))
                },
            )
        }
    }

    private fun createViewerSection(): LinearLayout {
        return PaafektScreenViews.createSectionCard(this).apply {
            addView(PaafektScreenViews.createSectionLabel(this@SettingsActivity, getString(R.string.settings_room_viewer)))
            addView(
                PaafektScreenViews.createToggleRow(
                    this@SettingsActivity,
                    getString(R.string.settings_auto_orbit),
                    getString(R.string.settings_auto_orbit_description),
                    prefs.getBoolean("auto_orbit_enabled", false),
                ) { isChecked ->
                    prefs.edit().putBoolean("auto_orbit_enabled", isChecked).apply()
                },
            )
            addView(
                PaafektScreenViews.createToggleRow(
                    this@SettingsActivity,
                    getString(R.string.settings_infinite_zoom),
                    getString(R.string.settings_infinite_zoom_description),
                    prefs.getBoolean("infinite_zoom_enabled", true),
                ) { isChecked ->
                    prefs.edit().putBoolean("infinite_zoom_enabled", isChecked).apply()
                },
            )
        }
    }

    private fun createRoomDefaultsSection(): LinearLayout {
        return PaafektScreenViews.createSectionCard(this).apply {
            addView(PaafektScreenViews.createSectionLabel(this@SettingsActivity, getString(R.string.settings_room_dimensions)))
            addView(PaafektScreenViews.createCaptionLabel(this@SettingsActivity, getString(R.string.settings_room_dimensions_footer)))
            addView(createDimensionSliderRow(
                label = getString(R.string.settings_width_label),
                initialValue = RoomDefaults.widthMeters(prefs),
                min = 2.0f,
                max = 10.0f,
                onValueChanged = { RoomDefaults.setWidthMeters(prefs, it) },
            ))
            addView(createDimensionSliderRow(
                label = getString(R.string.settings_height_label),
                initialValue = RoomDefaults.heightMeters(prefs),
                min = 2.0f,
                max = 5.0f,
                onValueChanged = { RoomDefaults.setHeightMeters(prefs, it) },
            ))
            addView(createDimensionSliderRow(
                label = getString(R.string.settings_depth_label),
                initialValue = RoomDefaults.depthMeters(prefs),
                min = 2.0f,
                max = 10.0f,
                onValueChanged = { RoomDefaults.setDepthMeters(prefs, it) },
            ))
        }
    }

    private fun createDeveloperSection(): LinearLayout {
        DebugLogger.init(this)
        return PaafektScreenViews.createSectionCard(this).apply {
            addView(PaafektScreenViews.createSectionLabel(this@SettingsActivity, getString(R.string.settings_developer)))
            addView(
                PaafektScreenViews.createToggleRow(
                    this@SettingsActivity,
                    getString(R.string.settings_debug_mode),
                    getString(R.string.settings_debug_mode_description),
                    DebugLogger.isDebugMode,
                ) { isChecked ->
                    DebugLogger.setDebugMode(isChecked)
                },
            )
            addView(PaafektScreenViews.createCaptionLabel(this@SettingsActivity, getString(R.string.settings_developer_footer)))
        }
    }

    private fun createLegalSection(): LinearLayout {
        return PaafektScreenViews.createSectionCard(this).apply {
            addView(PaafektScreenViews.createSectionLabel(this@SettingsActivity, getString(R.string.settings_legal)))
            addView(PaafektScreenViews.createLinkLabel(this@SettingsActivity, getString(R.string.settings_privacy_policy)) {
                openUrl("https://paafekt.com/privacy")
            })
            addView(PaafektScreenViews.createLinkLabel(this@SettingsActivity, getString(R.string.settings_terms_of_service)) {
                openUrl("https://paafekt.com/terms")
            })
            addView(PaafektScreenViews.createLinkLabel(this@SettingsActivity, getString(R.string.settings_support)) {
                startActivity(Intent(this@SettingsActivity, HelpActivity::class.java))
            })
            addView(PaafektScreenViews.createLinkLabel(this@SettingsActivity, getString(R.string.settings_credits)) {
                startActivity(Intent(this@SettingsActivity, CreditsActivity::class.java))
            })
            addView(PaafektScreenViews.createLinkLabel(this@SettingsActivity, getString(R.string.settings_licenses)) {
                startActivity(Intent(this@SettingsActivity, LicensesActivity::class.java))
            })
            addView(
                PaafektScreenViews.createCaptionLabel(
                    this@SettingsActivity,
                    getString(R.string.settings_legal_summary),
                ),
            )
        }
    }

    private fun createAboutSection(): LinearLayout {
        return PaafektScreenViews.createSectionCard(this).apply {
            addView(PaafektScreenViews.createSectionLabel(this@SettingsActivity, getString(R.string.profile_about)))
            addView(
                PaafektScreenViews.createSecondaryLabel(
                    this@SettingsActivity,
                    getString(
                        R.string.app_version_dynamic,
                        BuildConfig.VERSION_NAME,
                        BuildConfig.VERSION_CODE,
                    ),
                ),
            )
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
            setPadding(0, PaafektSpace.sm(this@SettingsActivity), 0, PaafektSpace.sm(this@SettingsActivity))

            val titleRow = LinearLayout(this@SettingsActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
            }

            val valueView = TextView(this@SettingsActivity).apply {
                text = getString(R.string.settings_dimension_value_meters, initialValue)
                textSize = 14f
                setTextColor(PaafektColors.accent)
            }

            titleRow.addView(
                PaafektScreenViews.createPrimaryLabel(this@SettingsActivity, label).apply {
                    layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                },
            )
            titleRow.addView(valueView)
            addView(titleRow)

            val seekBar = SeekBar(this@SettingsActivity).apply {
                this.max = maxSteps
                progress = (((initialValue - min) / step).toInt()).coerceIn(0, maxSteps)
                PaafektScreenViews.styleSeekBarGold(this)
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

    private fun showLogoutConfirmation() {
        AlertDialog.Builder(this, R.style.DarkDialogTheme)
            .setTitle(getString(R.string.settings_sign_out))
            .setMessage(getString(R.string.settings_sign_out_confirm))
            .setPositiveButton(getString(R.string.settings_sign_out)) { _, _ ->
                authManager.logout()
                navigateToLogin()
            }
            .setNegativeButton(getString(R.string.common_cancel), null)
            .show()
    }

    private fun showDeleteAccountConfirmation() {
        AlertDialog.Builder(this, R.style.DarkDialogTheme)
            .setTitle(R.string.settings_delete_account_confirm_title)
            .setMessage(R.string.settings_delete_account_confirm_message)
            .setPositiveButton(R.string.settings_delete_account) { _, _ ->
                authManager.deleteCurrentAccount { result ->
                    runOnUiThread {
                        result.onSuccess {
                            navigateToLogin()
                        }.onFailure { error ->
                            Toast.makeText(
                                this,
                                getString(
                                    R.string.settings_delete_account_failed,
                                    error.localizedMessage
                                        ?: getString(R.string.settings_delete_account_unknown_error),
                                ),
                                Toast.LENGTH_LONG,
                            ).show()
                        }
                    }
                }
            }
            .setNegativeButton(R.string.common_cancel, null)
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
            startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
        } catch (_: Exception) {
            Toast.makeText(this, R.string.settings_open_link_failed, Toast.LENGTH_SHORT).show()
        }
    }
}
