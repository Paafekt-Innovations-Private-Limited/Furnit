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
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
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
    private lateinit var countryButton: Button
    private lateinit var sendOtpButton: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var errorText: TextView

    private var selectedCountry: CountryCode = CountryCode.getDefaultCountry()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        authManager = AuthenticationManager.getInstance(this)

        // Check if already authenticated
        if (authManager.isAuthenticated) {
            navigateToMain()
            return
        }

        setupUI()
    }

    private fun setupUI() {
        val rootLayout = FrameLayout(this)

        // Gradient background (blue to purple like iOS)
        val gradientDrawable = GradientDrawable(
            GradientDrawable.Orientation.TL_BR,
            intArrayOf(
                Color.parseColor("#667eea"),
                Color.parseColor("#764ba2")
            )
        )
        rootLayout.background = gradientDrawable

        // Content card
        val cardLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.WHITE)
            setPadding(48, 48, 48, 48)
            elevation = 8f
        }

        val cardParams = FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply {
            setMargins(32, 100, 32, 32)
            gravity = Gravity.TOP
        }
        cardLayout.layoutParams = cardParams

        // App icon/logo placeholder
        val logoText = TextView(this).apply {
            text = "\uD83C\uDFE0" // House emoji
            textSize = 48f
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, 16)
        }
        cardLayout.addView(logoText)

        // Title
        val title = TextView(this).apply {
            text = "Welcome to Furnit"
            textSize = 24f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#333333"))
            gravity = Gravity.CENTER
        }
        cardLayout.addView(title)

        // Subtitle
        val subtitle = TextView(this).apply {
            text = "Sign in with your phone number"
            textSize = 14f
            setTextColor(Color.parseColor("#666666"))
            gravity = Gravity.CENTER
            setPadding(0, 8, 0, 32)
        }
        cardLayout.addView(subtitle)

        // Name input
        val nameLabel = TextView(this).apply {
            text = "Your Name"
            textSize = 14f
            setTextColor(Color.parseColor("#333333"))
            setPadding(0, 0, 0, 8)
        }
        cardLayout.addView(nameLabel)

        nameInput = EditText(this).apply {
            hint = "Enter your name"
            inputType = InputType.TYPE_TEXT_VARIATION_PERSON_NAME or InputType.TYPE_TEXT_FLAG_CAP_WORDS
            setBackgroundColor(Color.parseColor("#F5F5F5"))
            setPadding(24, 24, 24, 24)
            textSize = 16f
        }
        cardLayout.addView(nameInput, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { setMargins(0, 0, 0, 24) })

        // Phone number section
        val phoneLabel = TextView(this).apply {
            text = "Phone Number"
            textSize = 14f
            setTextColor(Color.parseColor("#333333"))
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
            setBackgroundColor(Color.parseColor("#E8E8E8"))
            setPadding(16, 16, 16, 16)
            setOnClickListener { showCountryPicker() }
        }
        phoneRow.addView(countryButton, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { setMargins(0, 0, 8, 0) })

        phoneInput = EditText(this).apply {
            hint = "Phone number"
            inputType = InputType.TYPE_CLASS_PHONE
            setBackgroundColor(Color.parseColor("#F5F5F5"))
            setPadding(24, 24, 24, 24)
            textSize = 16f
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
            setTextColor(Color.parseColor("#F44336"))
            gravity = Gravity.CENTER
            visibility = View.GONE
            setPadding(0, 0, 0, 16)
        }
        cardLayout.addView(errorText)

        // Send OTP button
        sendOtpButton = Button(this).apply {
            text = "Send Verification Code"
            textSize = 16f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#667eea"))
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
            text = "By continuing, you agree to our Terms of Service and Privacy Policy"
            textSize = 12f
            setTextColor(Color.parseColor("#999999"))
            gravity = Gravity.CENTER
            setPadding(0, 24, 0, 0)
        }
        cardLayout.addView(termsText)

        rootLayout.addView(cardLayout)
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

        val isValid = name.isNotEmpty() && phone.length >= 10
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

        Log.d(TAG, "Sending OTP to: $fullPhoneNumber")

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
