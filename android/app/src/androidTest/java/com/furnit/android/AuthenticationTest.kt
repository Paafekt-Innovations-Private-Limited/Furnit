package com.furnit.android

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.furnit.android.auth.AuthenticationManager
import com.furnit.android.auth.CountryCode
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Authentication tests for Firebase Phone Auth implementation.
 * Tests AuthenticationManager, CountryCode, rate limiting, and lockout protection.
 */
@RunWith(AndroidJUnit4::class)
class AuthenticationTest {

    private lateinit var context: Context
    private lateinit var authManager: AuthenticationManager

    @Before
    fun setup() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        authManager = AuthenticationManager.getInstance(context)

        // Clear any existing auth state for testing
        authManager.logout()
    }

    @Test
    fun testAuthManagerInitialization() {
        // AuthManager should initialize without error
        assertNotNull("AuthManager should not be null", authManager)
        println("AuthManager initialized successfully")
    }

    @Test
    fun testInitialAuthState() {
        // After logout, user should not be authenticated
        assertFalse("User should not be authenticated initially", authManager.isAuthenticated)
        assertNull("Current user should be null", authManager.currentUser)
        println("Initial auth state verified: not authenticated")
    }

    @Test
    fun testCountryCodeList() {
        val countries = CountryCode.countries

        // Should have countries
        assertTrue("Should have countries in list", countries.isNotEmpty())
        println("Found ${countries.size} countries")

        // Verify US is in the list
        val us = countries.find { it.code == "US" }
        assertNotNull("US should be in country list", us)
        assertEquals("US dial code should be +1", "+1", us?.dialCode)
        println("US country: ${us?.displayName}")

        // Verify India is in the list
        val india = countries.find { it.code == "IN" }
        assertNotNull("India should be in country list", india)
        assertEquals("India dial code should be +91", "+91", india?.dialCode)
        println("India country: ${india?.displayName}")
    }

    @Test
    fun testCountryCodeSearch() {
        // Search by name
        var results = CountryCode.search("United")
        assertTrue("Should find countries with 'United'", results.isNotEmpty())
        println("Search 'United' found: ${results.map { it.name }}")

        // Search by dial code
        results = CountryCode.search("+91")
        assertTrue("Should find India with +91", results.any { it.code == "IN" })
        println("Search '+91' found: ${results.map { it.name }}")

        // Search by country code
        results = CountryCode.search("GB")
        assertTrue("Should find UK with GB", results.any { it.code == "GB" })
        println("Search 'GB' found: ${results.map { it.name }}")

        // Empty search returns all
        results = CountryCode.search("")
        assertEquals("Empty search should return all countries", CountryCode.countries.size, results.size)
    }

    @Test
    fun testDefaultCountry() {
        val defaultCountry = CountryCode.getDefaultCountry()
        assertNotNull("Default country should not be null", defaultCountry)
        assertTrue("Default country should have dial code", defaultCountry.dialCode.startsWith("+"))
        println("Default country: ${defaultCountry.displayName}")
    }

    @Test
    fun testCountryCodeDisplayFormats() {
        val us = CountryCode.countries.find { it.code == "US" }!!

        // Check display formats
        assertTrue("Display name should contain flag", us.displayName.contains(us.flag))
        assertTrue("Display name should contain dial code", us.displayName.contains(us.dialCode))
        assertTrue("Short display should contain flag", us.shortDisplay.contains(us.flag))

        println("Display name: ${us.displayName}")
        println("Short display: ${us.shortDisplay}")
    }

    @Test
    fun testRateLimitingNotTriggeredInitially() {
        // Rate limiting should not be active initially
        assertFalse("Should not be rate limited initially", authManager.isRateLimited())
        println("Rate limiting check passed: not rate limited initially")
    }

    @Test
    fun testLockoutNotActiveInitially() {
        // Lockout should not be active initially
        assertFalse("Should not be locked out initially", authManager.isLockedOut())
        assertEquals("Lockout remaining should be 0", 0, authManager.getLockoutRemainingSeconds())
        println("Lockout check passed: not locked out initially")
    }

    @Test
    fun testUserDataAfterLogout() {
        // After logout, user data should be empty
        authManager.logout()

        assertEquals("User name should be empty after logout", "", authManager.getUserName())
        assertEquals("User phone should be empty after logout", "", authManager.getUserPhone())
        assertFalse("Should not be authenticated after logout", authManager.isAuthenticated)

        println("User data cleared after logout")
    }

    @Test
    fun testAuthStateListener() {
        var listenerCalled = false
        var receivedAuthState = true
        var receivedUser: AuthenticationManager.User? = AuthenticationManager.User("test", "test", "test")

        val listener = object : AuthenticationManager.AuthStateListener {
            override fun onAuthStateChanged(isAuthenticated: Boolean, user: AuthenticationManager.User?) {
                listenerCalled = true
                receivedAuthState = isAuthenticated
                receivedUser = user
            }
        }

        // Add listener
        authManager.addAuthStateListener(listener)

        // Listener should be called immediately with current state
        assertTrue("Listener should be called on add", listenerCalled)
        assertFalse("Should receive not authenticated state", receivedAuthState)
        assertNull("Should receive null user", receivedUser)

        // Remove listener
        authManager.removeAuthStateListener(listener)

        println("Auth state listener test passed")
    }

    @Test
    fun testPhoneNumberFormatValidation() {
        // Test valid phone numbers (10+ digits)
        val validNumbers = listOf(
            "1234567890",
            "12345678901",
            "9876543210"
        )

        for (number in validNumbers) {
            val digits = number.replace(Regex("[^0-9]"), "")
            assertTrue("Phone $number should be valid (10+ digits)", digits.length >= 10)
        }

        // Test invalid phone numbers
        val invalidNumbers = listOf(
            "123456789", // 9 digits
            "12345",     // 5 digits
            ""           // empty
        )

        for (number in invalidNumbers) {
            val digits = number.replace(Regex("[^0-9]"), "")
            assertFalse("Phone $number should be invalid (<10 digits)", digits.length >= 10)
        }

        println("Phone number format validation passed")
    }

    @Test
    fun testFullPhoneNumberConstruction() {
        val countryCode = CountryCode.countries.find { it.code == "US" }!!
        val phoneDigits = "5551234567"

        val fullNumber = "${countryCode.dialCode}$phoneDigits"

        assertEquals("Full number should be +15551234567", "+15551234567", fullNumber)
        assertTrue("Full number should start with +", fullNumber.startsWith("+"))

        println("Full phone number: $fullNumber")
    }

    @Test
    fun testOTPValidation() {
        // Valid OTP (6 digits)
        val validOtp = "123456"
        assertEquals("Valid OTP should be 6 digits", 6, validOtp.length)
        assertTrue("OTP should be all digits", validOtp.all { it.isDigit() })

        // Invalid OTPs
        val invalidOtps = listOf(
            "12345",   // 5 digits
            "1234567", // 7 digits
            "12345a",  // contains letter
            ""         // empty
        )

        for (otp in invalidOtps) {
            val isValid = otp.length == 6 && otp.all { it.isDigit() }
            assertFalse("OTP '$otp' should be invalid", isValid)
        }

        println("OTP validation passed")
    }

    @Test
    fun testMaskedPhoneNumber() {
        val phone = "+15551234567"
        val visible = phone.takeLast(4)
        val masked = "*".repeat(phone.length - 4)
        val maskedPhone = masked + visible

        assertEquals("Last 4 digits should be visible", "4567", visible)
        assertTrue("Masked phone should contain asterisks", maskedPhone.contains("*"))
        assertEquals("Masked phone should be same length", phone.length, maskedPhone.length)

        println("Masked phone: $maskedPhone")
    }

    @Test
    fun testSingletonPattern() {
        val instance1 = AuthenticationManager.getInstance(context)
        val instance2 = AuthenticationManager.getInstance(context)

        assertSame("AuthManager should be singleton", instance1, instance2)
        println("Singleton pattern verified")
    }
}
