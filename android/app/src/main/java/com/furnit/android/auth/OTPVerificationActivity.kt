package com.furnit.android.auth

import android.content.Intent
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.os.CountDownTimer
import android.text.Editable
import android.text.InputFilter
import android.text.InputType
import android.text.TextWatcher
import android.view.Gravity
import android.view.KeyEvent
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.content.res.ColorStateList
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.furnit.android.ContentActivity
import com.furnit.android.R
import com.furnit.android.theme.PaafektColors
import com.furnit.android.theme.PaafektDrawables
import com.furnit.android.utils.LogUtil

/**
 * OTP verification — 6-digit code.
 *
 * Note: We do **not** use Play Services SMS User Consent here. Firebase Phone Auth already
 * registers for [SMS_RETRIEVED]; starting SmsRetriever User Consent in parallel causes
 * broadcasts where the message body is null until consent, and Firebase’s internal receiver
 * (firebase-auth) can NPE on that path. Autofill / keyboard suggestions still work via
 * [setAutofillHints] and visible digit fields.
 */
class OTPVerificationActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "OTPVerification"
        const val EXTRA_PHONE_NUMBER = "phone_number"
        const val EXTRA_USER_NAME = "user_name"
        private const val OTP_LENGTH = 6
        private const val RESEND_COOLDOWN_SECONDS = 30
    }

    private lateinit var authManager: AuthenticationManager
    private lateinit var otpDigitInputs: Array<EditText>
    private lateinit var verifyButton: Button
    private lateinit var resendButton: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var errorText: TextView
    private lateinit var timerText: TextView

    private var phoneNumber: String = ""
    private var userName: String = ""
    private var resendTimer: CountDownTimer? = null
    private var canResend: Boolean = false
    private var pendingShowIme: Boolean = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        authManager = AuthenticationManager.getInstance(this)
        phoneNumber = intent.getStringExtra(EXTRA_PHONE_NUMBER) ?: ""
        userName = intent.getStringExtra(EXTRA_USER_NAME) ?: ""

        if (phoneNumber.isEmpty()) {
            finish()
            return
        }

        setupUI()
        startResendTimer()
        pendingShowIme = true
    }

    private fun dp(v: Float): Int = (v * resources.displayMetrics.density).toInt()

    private fun otpFieldBackground(hasFocus: Boolean): GradientDrawable {
        return GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            cornerRadius = dp(12f).toFloat()
            setColor(PaafektColors.surfaceHi)
            val strokePx = if (hasFocus) dp(2f) else dp(1f)
            val strokeColor = if (hasFocus) PaafektColors.accent else PaafektColors.hairline
            setStroke(strokePx, strokeColor)
        }
    }

    private fun refreshOtpBoxStyles() {
        otpDigitInputs.forEach { et ->
            et.background = otpFieldBackground(et.hasFocus())
        }
    }

    private fun setupUI() {
        val rootLayout = FrameLayout(this).apply {
            setBackgroundColor(PaafektColors.background)
        }

        val scrollContent = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(24f), dp(48f), dp(24f), dp(24f))
        }

        val headerIcon = ImageView(this).apply {
            setImageResource(R.drawable.ic_lock_shield)
            imageTintList = ColorStateList.valueOf(PaafektColors.accent)
            adjustViewBounds = true
            contentDescription = getString(R.string.otp_title)
        }
        scrollContent.addView(
            headerIcon,
            LinearLayout.LayoutParams(dp(50f), dp(50f)).apply {
                gravity = Gravity.CENTER_HORIZONTAL
                bottomMargin = dp(16f)
            },
        )

        val title = TextView(this).apply {
            text = getString(R.string.otp_title)
            textSize = 24f
            setTypeface(null, Typeface.BOLD)
            setTextColor(PaafektColors.textPrimary)
            gravity = Gravity.CENTER
        }
        scrollContent.addView(title, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
        ).apply { bottomMargin = dp(8f) })

        val subtitle = TextView(this).apply {
            text = getString(R.string.otp_subtitle)
            textSize = 15f
            setTextColor(PaafektColors.textSecondary)
            gravity = Gravity.CENTER
        }
        scrollContent.addView(subtitle, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
        ).apply { bottomMargin = dp(4f) })

        val phoneDisplay = TextView(this).apply {
            text = maskPhoneNumber(phoneNumber)
            textSize = 16f
            setTypeface(null, Typeface.BOLD)
            setTextColor(PaafektColors.textPrimary)
            gravity = Gravity.CENTER
        }
        scrollContent.addView(phoneDisplay, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
        ).apply { bottomMargin = dp(24f) })

        val cardLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = PaafektDrawables.secondaryButton()
            setPadding(dp(24f), dp(24f), dp(24f), dp(24f))
            elevation = 8f
        }

        val otpRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            importantForAutofill = View.IMPORTANT_FOR_AUTOFILL_YES
        }

        otpDigitInputs = Array(OTP_LENGTH) { index ->
            createOtpDigitInput(index)
        }

        otpDigitInputs.forEach { input ->
            otpRow.addView(
                input,
                LinearLayout.LayoutParams(dp(48f), dp(56f)).apply {
                    setMargins(dp(4f), 0, dp(4f), 0)
                },
            )
        }

        cardLayout.addView(
            otpRow,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply { bottomMargin = dp(20f) },
        )

        errorText = TextView(this).apply {
            textSize = 14f
            setTextColor(PaafektColors.danger)
            gravity = Gravity.CENTER
            visibility = View.GONE
            setPadding(0, 0, 0, dp(16f))
        }
        cardLayout.addView(errorText)

        verifyButton = Button(this).apply {
            text = getString(R.string.otp_verify)
            textSize = 16f
            setTypeface(null, Typeface.BOLD)
            setTextColor(PaafektColors.accentText)
            background = PaafektDrawables.primaryButton()
            setPadding(dp(24f), dp(24f), dp(24f), dp(24f))
            isEnabled = false
            alpha = 0.5f
            setOnClickListener { verifyOtp() }
        }
        cardLayout.addView(
            verifyButton,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        progressBar = ProgressBar(this).apply {
            visibility = View.GONE
            setPadding(0, dp(16f), 0, 0)
        }
        cardLayout.addView(
            progressBar,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply { gravity = Gravity.CENTER_HORIZONTAL },
        )

        timerText = TextView(this).apply {
            text = getString(R.string.otp_resend_in, RESEND_COOLDOWN_SECONDS)
            textSize = 14f
            setTextColor(PaafektColors.textSecondary)
            gravity = Gravity.CENTER
            setPadding(0, dp(24f), 0, 0)
        }
        cardLayout.addView(timerText)

        resendButton = TextView(this).apply {
            text = getString(R.string.otp_resend)
            textSize = 14f
            setTypeface(null, Typeface.BOLD)
            setTextColor(PaafektColors.accent)
            gravity = Gravity.CENTER
            setPadding(0, dp(8f), 0, 0)
            visibility = View.GONE
            setOnClickListener { resendOtp() }
        }
        cardLayout.addView(resendButton)

        scrollContent.addView(cardLayout, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
        ))

        rootLayout.addView(
            scrollContent,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )

        val backButton = TextView(this).apply {
            text = getString(R.string.otp_back)
            textSize = 16f
            setTextColor(PaafektColors.accent)
            setPadding(dp(16f), dp(48f), dp(16f), dp(16f))
            setOnClickListener { finish() }
        }
        rootLayout.addView(
            backButton,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply { gravity = Gravity.START or Gravity.TOP },
        )

        setContentView(rootLayout)
        refreshOtpBoxStyles()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (!hasFocus || !pendingShowIme || !::otpDigitInputs.isInitialized) return
        pendingShowIme = false
        otpDigitInputs[0].post {
            otpDigitInputs[0].requestFocus()
            val imm = getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager
            if (otpDigitInputs[0].windowToken != null) {
                imm.showSoftInput(otpDigitInputs[0], InputMethodManager.SHOW_IMPLICIT)
            }
        }
    }

    private fun createOtpDigitInput(index: Int): EditText {
        return EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER
            setRawInputType(InputType.TYPE_CLASS_NUMBER)
            imeOptions = if (index == OTP_LENGTH - 1) {
                EditorInfo.IME_ACTION_DONE
            } else {
                EditorInfo.IME_ACTION_NEXT
            }
            textSize = 22f
            gravity = Gravity.CENTER
            setTypeface(null, Typeface.BOLD)
            background = otpFieldBackground(false)
            setTextColor(ColorStateList.valueOf(PaafektColors.textPrimary))
            setHintTextColor(ColorStateList.valueOf(PaafektColors.textSecondary))
            highlightColor = PaafektColors.accent
            filters = arrayOf(InputFilter.LengthFilter(1))
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                importantForAutofill = View.IMPORTANT_FOR_AUTOFILL_YES
                if (index == 0) {
                    setAutofillHints("smsOTPCode")
                }
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val cursor = GradientDrawable().apply {
                    shape = GradientDrawable.RECTANGLE
                    setSize(dp(2f), dp(24f))
                    setColor(PaafektColors.accent)
                }
                textCursorDrawable = cursor
            }

            setOnFocusChangeListener { _, _ -> refreshOtpBoxStyles() }

            addTextChangedListener(object : TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
                override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
                override fun afterTextChanged(s: Editable?) {
                    if (s?.length == 1 && index < OTP_LENGTH - 1) {
                        otpDigitInputs[index + 1].requestFocus()
                    }
                    validateOtp()
                }
            })

            setOnKeyListener { _, keyCode, event ->
                if (keyCode == KeyEvent.KEYCODE_DEL && event.action == KeyEvent.ACTION_DOWN) {
                    if (text.isEmpty() && index > 0) {
                        otpDigitInputs[index - 1].apply {
                            requestFocus()
                            text.clear()
                        }
                        return@setOnKeyListener true
                    }
                }
                false
            }
        }
    }

    private fun maskPhoneNumber(phone: String): String {
        if (phone.length < 4) return phone
        val visible = phone.takeLast(4)
        val masked = "*".repeat(phone.length - 4)
        return masked + visible
    }

    private fun validateOtp() {
        val otp = getEnteredOtp()
        val isComplete = otp.length == OTP_LENGTH
        verifyButton.isEnabled = isComplete
        verifyButton.alpha = if (isComplete) 1.0f else 0.5f

        if (isComplete) {
            verifyOtp()
        }
    }

    private fun getEnteredOtp(): String {
        return otpDigitInputs.joinToString("") { it.text.toString() }
    }

    private fun clearOtp() {
        otpDigitInputs.forEach { it.text.clear() }
        otpDigitInputs[0].requestFocus()
    }

    private fun verifyOtp() {
        val otp = getEnteredOtp()
        if (otp.length != OTP_LENGTH) return

        LogUtil.d(TAG, "Verifying OTP: $otp")

        errorText.visibility = View.GONE
        progressBar.visibility = View.VISIBLE
        verifyButton.isEnabled = false

        authManager.verifyOTP(
            otp = otp,
            name = userName,
            phoneNumber = phoneNumber,
            onSuccess = {
                progressBar.visibility = View.GONE
                navigateToMain()
            },
            onError = { error ->
                progressBar.visibility = View.GONE
                verifyButton.isEnabled = true
                errorText.text = error
                errorText.visibility = View.VISIBLE
                clearOtp()
            },
        )
    }

    private fun startResendTimer() {
        canResend = false
        timerText.visibility = View.VISIBLE
        resendButton.visibility = View.GONE

        resendTimer?.cancel()
        resendTimer = object : CountDownTimer(RESEND_COOLDOWN_SECONDS * 1000L, 1000) {
            override fun onTick(millisUntilFinished: Long) {
                val seconds = (millisUntilFinished / 1000).toInt()
                timerText.text = getString(R.string.otp_resend_in, seconds)
            }

            override fun onFinish() {
                canResend = true
                timerText.visibility = View.GONE
                resendButton.visibility = View.VISIBLE
            }
        }.start()
    }

    private fun resendOtp() {
        if (!canResend) return

        LogUtil.d(TAG, "Resending OTP to: $phoneNumber")

        resendButton.isEnabled = false
        progressBar.visibility = View.VISIBLE

        authManager.resendOTP(
            phoneNumber = phoneNumber,
            activity = this,
            onCodeSent = {
                progressBar.visibility = View.GONE
                resendButton.isEnabled = true
                Toast.makeText(this, R.string.otp_code_sent, Toast.LENGTH_SHORT).show()
                startResendTimer()
            },
            onError = { error ->
                progressBar.visibility = View.GONE
                resendButton.isEnabled = true
                errorText.text = error
                errorText.visibility = View.VISIBLE
            },
        )
    }

    private fun navigateToMain() {
        val intent = Intent(this, ContentActivity::class.java)
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        startActivity(intent)
        finish()
    }

    override fun onDestroy() {
        super.onDestroy()
        resendTimer?.cancel()
    }
}
