package com.furnit.android.auth

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AuthenticationManagerDebugSettingsTest {

    @Test
    fun testNumberAllowsDebugSettingsAcrossDisplayFormatting() {
        assertTrue(AuthenticationManager.isDebugSettingsPhoneNumber("+16505553434"))
        assertTrue(AuthenticationManager.isDebugSettingsPhoneNumber("+1 650-555-3434"))
    }

    @Test
    fun otherNumbersDoNotAllowDebugSettings() {
        assertFalse(AuthenticationManager.isDebugSettingsPhoneNumber(null))
        assertFalse(AuthenticationManager.isDebugSettingsPhoneNumber(""))
        assertFalse(AuthenticationManager.isDebugSettingsPhoneNumber("+16505553435"))
        assertFalse(AuthenticationManager.isDebugSettingsPhoneNumber("+916505553434"))
        assertFalse(AuthenticationManager.isDebugSettingsPhoneNumber("(650) 555-3434"))
    }
}
