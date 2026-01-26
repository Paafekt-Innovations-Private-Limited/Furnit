package com.furnit.android.auth

import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.CountDownTimer
import android.text.Editable
import android.text.InputFilter
import android.text.InputType
import android.text.TextWatcher
import android.util.Log
import android.view.Gravity
import android.view.KeyEvent
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.content.res.ColorStateList
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import com.furnit.android.ContentActivity

/**
 * OTPVerificationActivity - 6-digit OTP verification screen
 * Matches iOS OTPVerificationView.swift
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
    }

    private fun setupUI() {
        val rootLayout = FrameLayout(this)

        // Gradient background
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

        // Back button
        val backButton = TextView(this).apply {
            text = "< Back"
            textSize = 16f
            setTextColor(Color.parseColor("#667eea"))
            setPadding(0, 0, 0, 24)
            setOnClickListener { finish() }
        }
        cardLayout.addView(backButton)

        // Title
        val title = TextView(this).apply {
            text = "Verify Phone"
            textSize = 24f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#333333"))
            gravity = Gravity.CENTER
        }
        cardLayout.addView(title)

        // Subtitle with phone number
        val maskedPhone = maskPhoneNumber(phoneNumber)
        val subtitle = TextView(this).apply {
            text = "Enter the 6-digit code sent to\n$maskedPhone"
            textSize = 14f
            setTextColor(Color.parseColor("#666666"))
            gravity = Gravity.CENTER
            setPadding(0, 8, 0, 32)
        }
        cardLayout.addView(subtitle)

        // OTP input row
        val otpRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }

        otpDigitInputs = Array(OTP_LENGTH) { index ->
            createOtpDigitInput(index)
        }

        otpDigitInputs.forEach { input ->
            otpRow.addView(input, LinearLayout.LayoutParams(
                56, 64
            ).apply { setMargins(4, 0, 4, 0) })
        }

        cardLayout.addView(otpRow, LinearLayout.LayoutParams(
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

        // Verify button
        verifyButton = Button(this).apply {
            text = "Verify"
            textSize = 16f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#667eea"))
            setPadding(24, 24, 24, 24)
            isEnabled = false
            alpha = 0.5f
            setOnClickListener { verifyOtp() }
        }
        cardLayout.addView(verifyButton, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ))

        // Progress bar
        progressBar = ProgressBar(this).apply {
            visibility = View.GONE
            setPadding(0, 16, 0, 0)
        }
        cardLayout.addView(progressBar, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { gravity = Gravity.CENTER_HORIZONTAL })

        // Timer text
        timerText = TextView(this).apply {
            text = "Resend code in ${RESEND_COOLDOWN_SECONDS}s"
            textSize = 14f
            setTextColor(Color.parseColor("#999999"))
            gravity = Gravity.CENTER
            setPadding(0, 24, 0, 0)
        }
        cardLayout.addView(timerText)

        // Resend button
        resendButton = TextView(this).apply {
            text = "Resend Code"
            textSize = 14f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#667eea"))
            gravity = Gravity.CENTER
            setPadding(0, 8, 0, 0)
            visibility = View.GONE
            setOnClickListener { resendOtp() }
        }
        cardLayout.addView(resendButton)

        rootLayout.addView(cardLayout)
        setContentView(rootLayout)

        // Focus first input
        otpDigitInputs[0].requestFocus()
    }

    private fun createOtpDigitInput(index: Int): EditText {
        return EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER
            textSize = 24f
            gravity = Gravity.CENTER
            setTypeface(null, Typeface.BOLD)
            setBackgroundColor(Color.parseColor("#F5F5F5"))
            // Force black text color with ColorStateList to override theme
            setTextColor(ColorStateList.valueOf(Color.parseColor("#000000")))
            setHintTextColor(ColorStateList.valueOf(Color.parseColor("#999999")))
            highlightColor = Color.parseColor("#667eea")
            filters = arrayOf(InputFilter.LengthFilter(1))
            imeOptions = if (index == OTP_LENGTH - 1) EditorInfo.IME_ACTION_DONE else EditorInfo.IME_ACTION_NEXT

            addTextChangedListener(object : TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
                override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
                override fun afterTextChanged(s: Editable?) {
                    if (s?.length == 1 && index < OTP_LENGTH - 1) {
                        // Move to next input
                        otpDigitInputs[index + 1].requestFocus()
                    }
                    validateOtp()
                }
            })

            setOnKeyListener { _, keyCode, event ->
                if (keyCode == KeyEvent.KEYCODE_DEL && event.action == KeyEvent.ACTION_DOWN) {
                    if (text.isEmpty() && index > 0) {
                        // Move to previous input on backspace when empty
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

        // Auto-verify when all digits entered
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

        Log.d(TAG, "Verifying OTP: $otp")

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
            }
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
                timerText.text = "Resend code in ${seconds}s"
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

        Log.d(TAG, "Resending OTP to: $phoneNumber")

        resendButton.isEnabled = false
        progressBar.visibility = View.VISIBLE

        authManager.resendOTP(
            phoneNumber = phoneNumber,
            activity = this,
            onCodeSent = {
                progressBar.visibility = View.GONE
                resendButton.isEnabled = true
                Toast.makeText(this, "Code sent!", Toast.LENGTH_SHORT).show()
                startResendTimer()
            },
            onError = { error ->
                progressBar.visibility = View.GONE
                resendButton.isEnabled = true
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

    override fun onDestroy() {
        super.onDestroy()
        resendTimer?.cancel()
    }
}
