package com.furnit.android.auth

import org.junit.Assert.assertEquals
import org.junit.Test

class OtpCodeInputTest {

    @Test
    fun extractsSixDigitCodeFromKeyboardOrSmsSuggestion() {
        assertEquals("123456", OtpCodeInput.asciiDigits("123456"))
        assertEquals("123456", OtpCodeInput.asciiDigits("Your Paafekt code is 123456"))
    }

    @Test
    fun rejectsNonAsciiNumeralsAndLimitsUnexpectedExtraDigits() {
        assertEquals("", OtpCodeInput.asciiDigits(null))
        assertEquals("123456", OtpCodeInput.asciiDigits("123456789"))
        assertEquals("", OtpCodeInput.asciiDigits("१२३४५६"))
    }

    @Test
    fun extractsExactSixDigitCodeFromConsentedSms() {
        assertEquals("123456", OtpCodeInput.fromSmsMessage("Your Paafekt code is 123456"))
        assertEquals("654321", OtpCodeInput.fromSmsMessage("<#> 654321 is your verification code\nAbCdEfGhijk"))
        assertEquals(null, OtpCodeInput.fromSmsMessage("Reference 1234567"))
        assertEquals(null, OtpCodeInput.fromSmsMessage(null))
    }
}
