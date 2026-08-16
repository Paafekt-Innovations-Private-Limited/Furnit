package com.furnit.android.auth

/** Pure OTP normalization shared by manual, paste, and SMS User Consent input. */
internal object OtpCodeInput {
    const val LENGTH = 6
    private val SIX_DIGIT_SMS_CODE = Regex("(?<![0-9])([0-9]{6})(?![0-9])")

    fun asciiDigits(candidate: CharSequence?): String {
        return candidate?.toString().orEmpty()
            .filter { it in '0'..'9' }
            .take(LENGTH)
    }

    fun fromSmsMessage(message: CharSequence?): String? {
        return SIX_DIGIT_SMS_CODE.find(message?.toString().orEmpty())?.groupValues?.get(1)
    }
}
