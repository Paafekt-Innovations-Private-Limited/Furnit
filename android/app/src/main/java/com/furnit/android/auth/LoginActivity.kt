package com.furnit.android.auth

import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.text.Editable
import android.text.InputFilter
import android.text.InputType
import android.text.TextWatcher
import com.furnit.android.theme.PaafektColors
import com.furnit.android.theme.PaafektDrawables
import com.furnit.android.utils.LogUtil
import com.furnit.android.utils.WindowInsetsUtil
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.content.res.ColorStateList
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.furnit.android.ContentActivity
import com.furnit.android.R

/**
 * LoginActivity - Phone number login with country code selection
 * Matches iOS LoginView.swift
 */
class LoginActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "LoginActivity"
    }

    private lateinit var authManager: AuthenticationManager
    private lateinit var nameInput: EditText
    private lateinit var phoneInput: EditText
    private lateinit var nameHint: TextView
    private lateinit var countryButton: Button
    private lateinit var sendOtpButton: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var errorText: TextView

    private lateinit var selectedCountry: CountryCode

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        authManager = AuthenticationManager.getInstance(this)

        // Initialize country code based on SIM/network/locale
        selectedCountry = CountryCode.getDefaultCountry(this)

        // Check if already authenticated
        if (authManager.isAuthenticated) {
            navigateToMain()
            return
        }

        setupUI()
    }

    private fun setupUI() {
        val density = resources.displayMetrics.density
        fun dp(value: Int): Int = (value * density).toInt()

        val rootLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(PaafektColors.background)
            setPadding(dp(24), dp(24), dp(24), dp(24))
            // Edge-to-edge (targetSdk 35+) draws behind the system bars; add the real
            // status/navigation bar insets so content clears the notification bar.
            WindowInsetsUtil.applySystemBarInsetsAsPadding(this)
        }

        val logoView = ImageView(this).apply {
            setImageResource(R.drawable.paafekt_login_mark)
            adjustViewBounds = true
            scaleType = ImageView.ScaleType.FIT_CENTER
            contentDescription = getString(R.string.app_name)
        }
        rootLayout.addView(
            logoView,
            LinearLayout.LayoutParams(dp(96), dp(96)).apply {
                gravity = Gravity.CENTER_HORIZONTAL
                bottomMargin = dp(16)
            }
        )

        val appNameView = TextView(this).apply {
            text = getString(R.string.app_name)
            textSize = 24f
            setTypeface(null, Typeface.BOLD)
            setTextColor(PaafektColors.textPrimary)
            gravity = Gravity.CENTER
        }
        rootLayout.addView(appNameView, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = dp(4) })

        val taglineView = TextView(this).apply {
            text = getString(R.string.app_tagline)
            textSize = 15f
            setTextColor(PaafektColors.textSecondary)
            gravity = Gravity.CENTER
        }
        rootLayout.addView(taglineView, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = dp(24) })

        // Sign-in card
        val cardLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = PaafektDrawables.secondaryButton()
            setPadding(dp(24), dp(24), dp(24), dp(24))
            elevation = 8f
        }
        val nameLabel = TextView(this).apply {
            text = getString(R.string.login_your_name)
            textSize = 14f
            setTextColor(PaafektColors.textPrimary)
            setPadding(0, 0, 0, 8)
        }
        cardLayout.addView(nameLabel)

        nameInput = EditText(this).apply {
            hint = getString(R.string.login_enter_name)
            inputType = InputType.TYPE_TEXT_VARIATION_PERSON_NAME or InputType.TYPE_TEXT_FLAG_CAP_WORDS
            setBackgroundColor(PaafektColors.surfaceHi)
            setPadding(24, 24, 24, 24)
            textSize = 16f
            setTextColor(ColorStateList.valueOf(PaafektColors.textPrimary))
            setHintTextColor(ColorStateList.valueOf(PaafektColors.textSecondary))
        }
        cardLayout.addView(nameInput, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { setMargins(0, 0, 0, 4) })

        nameHint = TextView(this).apply {
            text = getString(R.string.login_name_hint)
            textSize = 12f
            setTextColor(PaafektColors.textSecondary)
            setPadding(4, 0, 4, 0)
        }
        cardLayout.addView(nameHint, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { setMargins(0, 0, 0, 24) })

        // Phone number section
        val phoneLabel = TextView(this).apply {
            text = getString(R.string.login_phone_number)
            textSize = 14f
            setTextColor(PaafektColors.textPrimary)
            setPadding(0, 0, 0, 8)
        }
        cardLayout.addView(phoneLabel)

        // Country code + phone input row
        val phoneRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        countryButton = Button(this).apply {
            text = selectedCountry.shortDisplay
            textSize = 14f
            setTextColor(ColorStateList.valueOf(PaafektColors.textPrimary))
            setBackgroundColor(PaafektColors.surfaceHi)
            setPadding(16, 16, 16, 16)
            setOnClickListener { showCountryPicker() }
        }
        phoneRow.addView(countryButton, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { setMargins(0, 0, 8, 0) })

        phoneInput = EditText(this).apply {
            hint = getString(R.string.login_phone_placeholder)
            inputType = InputType.TYPE_CLASS_PHONE
            setBackgroundColor(PaafektColors.surfaceHi)
            setPadding(24, 24, 24, 24)
            textSize = 16f
            setTextColor(ColorStateList.valueOf(PaafektColors.textPrimary))
            setHintTextColor(ColorStateList.valueOf(PaafektColors.textSecondary))
            filters = arrayOf(InputFilter.LengthFilter(15))
            imeOptions = EditorInfo.IME_ACTION_DONE
        }
        phoneRow.addView(phoneInput, LinearLayout.LayoutParams(
            0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f
        ))

        cardLayout.addView(phoneRow, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { setMargins(0, 0, 0, 24) })

        // Error text
        errorText = TextView(this).apply {
            textSize = 14f
            setTextColor(PaafektColors.danger)
            gravity = Gravity.CENTER
            visibility = View.GONE
            setPadding(0, 0, 0, 16)
        }
        cardLayout.addView(errorText)

        // Send OTP button
        sendOtpButton = Button(this).apply {
            text = getString(R.string.login_send_code)
            textSize = 16f
            setTypeface(null, Typeface.BOLD)
            setTextColor(PaafektColors.accentText)
            background = PaafektDrawables.primaryButton()
            setPadding(24, 24, 24, 24)
            isEnabled = false
            setOnClickListener { sendOtp() }
        }
        cardLayout.addView(sendOtpButton, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ))

        // Progress bar (hidden by default)
        progressBar = ProgressBar(this).apply {
            visibility = View.GONE
            setPadding(0, 16, 0, 0)
        }
        cardLayout.addView(progressBar, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { gravity = Gravity.CENTER_HORIZONTAL })

        // Terms text
        val termsText = TextView(this).apply {
            text = getString(R.string.login_terms_agree)
            textSize = 12f
            setTextColor(PaafektColors.textSecondary)
            gravity = Gravity.CENTER
            setPadding(0, 24, 0, 0)
        }
        cardLayout.addView(termsText)

        rootLayout.addView(cardLayout, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ))
        setContentView(rootLayout)

        // Add text watchers for validation
        val textWatcher = object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                validateInputs()
            }
        }
        nameInput.addTextChangedListener(textWatcher)
        phoneInput.addTextChangedListener(textWatcher)
    }

    private fun validateInputs() {
        val name = nameInput.text.toString().trim()
        val phone = phoneInput.text.toString().replace(Regex("[^0-9]"), "")
        val nameValid = com.furnit.android.util.DisplayNameValidation.isValid(name)

        if (::nameHint.isInitialized) {
            if (name.isNotEmpty() && !nameValid) {
                nameHint.text = getString(R.string.login_invalid_name)
                nameHint.setTextColor(PaafektColors.accent)
            } else {
                nameHint.text = getString(R.string.login_name_hint)
                nameHint.setTextColor(PaafektColors.textSecondary)
            }
        }

        val isValid = nameValid && phone.length >= 10
        sendOtpButton.isEnabled = isValid
        sendOtpButton.alpha = if (isValid) 1.0f else 0.5f
    }

    private fun showCountryPicker() {
        val countries = CountryCode.countries
        val countryNames = countries.map { it.displayName }.toTypedArray()

        AlertDialog.Builder(this)
            .setTitle("Select Country")
            .setItems(countryNames) { _, which ->
                selectedCountry = countries[which]
                countryButton.text = selectedCountry.shortDisplay
            }
            .show()
    }

    private fun sendOtp() {
        val name = nameInput.text.toString().trim()
        val phoneDigits = phoneInput.text.toString().replace(Regex("[^0-9]"), "")
        val fullPhoneNumber = "${selectedCountry.dialCode}$phoneDigits"

        if (!com.furnit.android.util.DisplayNameValidation.isValid(name) || phoneDigits.length < 10) {
            errorText.text = getString(R.string.login_validation_error)
            errorText.visibility = View.VISIBLE
            return
        }

        LogUtil.d(TAG, "Sending OTP to: $fullPhoneNumber")

        // Hide error, show progress
        errorText.visibility = View.GONE
        progressBar.visibility = View.VISIBLE
        sendOtpButton.isEnabled = false

        authManager.sendOTP(
            phoneNumber = fullPhoneNumber,
            activity = this,
            onCodeSent = {
                progressBar.visibility = View.GONE
                sendOtpButton.isEnabled = true

                // Navigate to OTP verification
                val intent = Intent(this, OTPVerificationActivity::class.java).apply {
                    putExtra(OTPVerificationActivity.EXTRA_PHONE_NUMBER, fullPhoneNumber)
                    putExtra(OTPVerificationActivity.EXTRA_USER_NAME, name)
                }
                startActivity(intent)
            },
            onError = { error ->
                progressBar.visibility = View.GONE
                sendOtpButton.isEnabled = true
                errorText.text = error
                errorText.visibility = View.VISIBLE
            }
        )
    }

    private fun navigateToMain() {
        val intent = Intent(this, ContentActivity::class.java)
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        startActivity(intent)
        finish()
    }

    override fun onResume() {
        super.onResume()
        // Check if user authenticated while we were away
        if (authManager.isAuthenticated) {
            navigateToMain()
        }
    }
}
