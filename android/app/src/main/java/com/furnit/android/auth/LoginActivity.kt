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
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
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

        // Handle insets ourselves so the keyboard (IME) inset is delivered to our
        // listener under enforced edge-to-edge; otherwise the keypad overlaps the fields.
        WindowCompat.setDecorFitsSystemWindows(window, false)

        authManager = AuthenticationManager.getInstance(this)

        // Prefer the current mobile network/SIM country, then fall back to the app locale.
        // This keeps an English (UK) language choice from incorrectly defaulting an Indian SIM to +44.
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

        // Scrolls when the keyboard covers the fields. System/IME insets are applied after
        // setContentView so the title clears the status bar and the Send button stays above
        // the keypad.
        val scrollView = ScrollView(this).apply {
            setBackgroundColor(PaafektColors.background)
            isFillViewport = true
            overScrollMode = View.OVER_SCROLL_NEVER
            // Base top padding; status-bar inset is added by WindowInsetsUtil.
            setPadding(0, dp(16), 0, dp(24))
            clipToPadding = false
        }

        val contentColumn = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(dp(16), dp(24), dp(16), dp(32))
        }

        // Premium Paafekt mark — centered above the sign-in card
        val logoView = ImageView(this).apply {
            setImageResource(R.drawable.paafekt_login_mark)
            adjustViewBounds = true
            scaleType = ImageView.ScaleType.FIT_CENTER
            contentDescription = getString(R.string.app_name)
        }
        contentColumn.addView(
            logoView,
            LinearLayout.LayoutParams(dp(96), dp(96)).apply {
                gravity = Gravity.CENTER_HORIZONTAL
                bottomMargin = dp(12)
            }
        )

        val appNameView = TextView(this).apply {
            text = getString(R.string.app_name)
            textSize = 24f
            setTypeface(null, Typeface.BOLD)
            setTextColor(PaafektColors.textPrimary)
            gravity = Gravity.CENTER
        }
        contentColumn.addView(appNameView, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = dp(12) })

        val taglineView = TextView(this).apply {
            text = getString(R.string.app_tagline)
            textSize = 15f
            setTextColor(PaafektColors.textSecondary)
            gravity = Gravity.CENTER
        }
        contentColumn.addView(taglineView, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = dp(24) })

        // Sign-in card — iOS paafektCardSurface(): surface fill, control radius, hairline stroke
        val cardLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = PaafektDrawables.secondaryButton()
            setPadding(dp(24), dp(24), dp(24), dp(24))
        }

        // Name field group (label + input + hint), iOS spacing 8
        val nameLabel = makeFieldLabel(getString(R.string.login_your_name), R.drawable.ic_person)
        cardLayout.addView(nameLabel, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = dp(8) })

        nameInput = EditText(this).apply {
            hint = getString(R.string.login_enter_name)
            inputType = InputType.TYPE_TEXT_VARIATION_PERSON_NAME or InputType.TYPE_TEXT_FLAG_CAP_WORDS
            background = PaafektDrawables.fieldSurface()
            setPadding(dp(16), dp(16), dp(16), dp(16))
            textSize = 16f
            setTextColor(ColorStateList.valueOf(PaafektColors.textPrimary))
            setHintTextColor(ColorStateList.valueOf(PaafektColors.textSecondary))
        }
        cardLayout.addView(nameInput, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = dp(8) })

        nameHint = TextView(this).apply {
            text = getString(R.string.login_name_hint)
            textSize = 12f
            setTextColor(PaafektColors.textSecondary)
        }
        cardLayout.addView(nameHint, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = dp(20) })

        // Phone field group (label + [country | number] row), iOS spacing 8
        val phoneLabel = makeFieldLabel(getString(R.string.login_phone_number), R.drawable.ic_phone)
        cardLayout.addView(phoneLabel, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = dp(8) })

        val phoneRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        countryButton = Button(this).apply {
            text = selectedCountry.shortDisplay
            textSize = 14f
            isAllCaps = false
            setTypeface(null, Typeface.NORMAL)
            setTextColor(ColorStateList.valueOf(PaafektColors.textPrimary))
            background = PaafektDrawables.fieldSurface()
            setPadding(dp(12), dp(14), dp(12), dp(14))
            stateListAnimator = null
            elevation = 0f
            compoundDrawablePadding = dp(4)
            setLeadingOrTrailingIcon(R.drawable.ic_chevron_down, dp(14), PaafektColors.textPrimary, trailing = true)
            setOnClickListener { showCountryPicker() }
        }
        phoneRow.addView(countryButton, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { setMargins(0, 0, dp(8), 0) })

        phoneInput = EditText(this).apply {
            hint = getString(R.string.login_phone_placeholder)
            inputType = InputType.TYPE_CLASS_PHONE
            background = PaafektDrawables.fieldSurface()
            setPadding(dp(16), dp(16), dp(16), dp(16))
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
        ).apply { bottomMargin = dp(20) })

        // Error text (inline) — hidden until validation fails
        errorText = TextView(this).apply {
            textSize = 14f
            setTextColor(PaafektColors.danger)
            gravity = Gravity.CENTER
            visibility = View.GONE
        }
        cardLayout.addView(errorText, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = dp(12) })

        // Send OTP button — primary control with a paperplane icon
        sendOtpButton = Button(this).apply {
            text = getString(R.string.login_send_code)
            textSize = 17f
            isAllCaps = false
            setTypeface(null, Typeface.BOLD)
            setTextColor(PaafektColors.accentText)
            background = PaafektDrawables.primaryButton()
            setPadding(dp(16), dp(14), dp(16), dp(14))
            stateListAnimator = null
            elevation = 0f
            compoundDrawablePadding = dp(8)
            setLeadingOrTrailingIcon(R.drawable.ic_send, dp(18), PaafektColors.accentText, trailing = false)
            isEnabled = false
            alpha = 0.5f
            setOnClickListener { sendOtp() }
        }
        cardLayout.addView(sendOtpButton, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = dp(20) })

        // Progress bar (hidden by default)
        progressBar = ProgressBar(this).apply {
            visibility = View.GONE
        }
        cardLayout.addView(progressBar, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply {
            gravity = Gravity.CENTER_HORIZONTAL
            bottomMargin = dp(12)
        })

        // Terms hint below the button
        val termsText = TextView(this).apply {
            text = getString(R.string.login_terms_agree)
            textSize = 12f
            setTextColor(PaafektColors.textSecondary)
            gravity = Gravity.CENTER
        }
        cardLayout.addView(termsText, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ))

        contentColumn.addView(cardLayout, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ))
        // WRAP_CONTENT so the form can scroll above the keyboard (MATCH_PARENT + center
        // kept the Send button under the IME).
        scrollView.addView(
            contentColumn,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply { gravity = Gravity.CENTER_VERTICAL },
        )
        setContentView(scrollView)

        WindowInsetsUtil.applyImeAwareInsetsAsPadding(scrollView) { sendOtpButton }
        ViewCompat.requestApplyInsets(scrollView)

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

        val bringSendButtonAboveKeyboard = View.OnFocusChangeListener { _, hasFocus ->
            if (hasFocus) {
                WindowInsetsUtil.bringViewAboveIme(sendOtpButton)
            }
        }
        nameInput.onFocusChangeListener = bringSendButtonAboveKeyboard
        phoneInput.onFocusChangeListener = bringSendButtonAboveKeyboard
    }

    /** Caption-style field label with a leading tinted icon — matches iOS `Label(..., systemImage:)`. */
    private fun makeFieldLabel(text: String, iconRes: Int): TextView {
        val density = resources.displayMetrics.density
        return TextView(this).apply {
            this.text = text
            textSize = 12f
            setTextColor(PaafektColors.textSecondary)
            compoundDrawablePadding = (6 * density).toInt()
            setLeadingOrTrailingIcon(iconRes, (14 * density).toInt(), PaafektColors.textSecondary, trailing = false)
        }
    }

    /** Sets a fixed-size, tinted compound drawable at the start (or end) of a TextView. */
    private fun TextView.setLeadingOrTrailingIcon(iconRes: Int, sizePx: Int, tint: Int, trailing: Boolean) {
        val icon = androidx.core.content.ContextCompat.getDrawable(context, iconRes)?.mutate() ?: return
        icon.setBounds(0, 0, sizePx, sizePx)
        icon.setTint(tint)
        if (trailing) {
            setCompoundDrawables(null, null, icon, null)
        } else {
            setCompoundDrawables(icon, null, null, null)
        }
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
        val countryNames = countries.map { it.localizedDisplayName(this) }.toTypedArray()

        AlertDialog.Builder(this)
            .setTitle(R.string.country_picker_title)
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
