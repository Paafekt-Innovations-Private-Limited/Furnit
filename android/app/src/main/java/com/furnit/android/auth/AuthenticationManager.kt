package com.furnit.android.auth

import android.app.Activity
import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import com.google.firebase.FirebaseException
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.PhoneAuthCredential
import com.google.firebase.auth.PhoneAuthOptions
import com.google.firebase.auth.PhoneAuthProvider
import java.util.concurrent.TimeUnit

/**
 * AuthenticationManager - Handles Firebase Phone + OTP authentication
 * Matches iOS AuthenticationManager.swift implementation
 */
class AuthenticationManager private constructor(context: Context) {

    companion object {
        private const val TAG = "AuthManager"
        private const val PREFS_NAME = "furnit_auth"
        private const val KEY_IS_AUTHENTICATED = "isAuthenticated"
        private const val KEY_USER_ID = "userId"
        private const val KEY_USER_NAME = "userName"
        private const val KEY_USER_PHONE = "userPhone"
        private const val KEY_OTP_ATTEMPTS = "otpAttempts"
        private const val KEY_OTP_REQUESTS = "otpRequests"
        private const val KEY_LOCKOUT_TIME = "lockoutTime"
        private const val KEY_LAST_OTP_REQUEST_TIME = "lastOtpRequestTime"

        private const val MAX_OTP_ATTEMPTS = 5
        private const val MAX_OTP_REQUESTS_PER_HOUR = 5
        private const val LOCKOUT_DURATION_MS = 30 * 60 * 1000L // 30 minutes
        private const val HOUR_MS = 60 * 60 * 1000L

        // Set to true to bypass OTP authentication (for development/testing)
        const val BYPASS_AUTH = true

        @Volatile
        private var instance: AuthenticationManager? = null

        fun getInstance(context: Context): AuthenticationManager {
            return instance ?: synchronized(this) {
                instance ?: AuthenticationManager(context.applicationContext).also { instance = it }
            }
        }
    }

    private val auth: FirebaseAuth = FirebaseAuth.getInstance()
    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private var verificationId: String? = null
    private var resendToken: PhoneAuthProvider.ForceResendingToken? = null

    // Observable state
    var isAuthenticated: Boolean = false
        private set
    var currentUser: User? = null
        private set
    var isLoading: Boolean = false
        private set
    var errorMessage: String? = null
        private set

    // Listeners for state changes
    private val authListeners = mutableListOf<AuthStateListener>()

    interface AuthStateListener {
        fun onAuthStateChanged(isAuthenticated: Boolean, user: User?)
    }

    data class User(
        val id: String,
        val name: String,
        val phoneNumber: String
    )

    init {
        checkAuthenticationStatus()
    }

    fun addAuthStateListener(listener: AuthStateListener) {
        authListeners.add(listener)
        listener.onAuthStateChanged(isAuthenticated, currentUser)
    }

    fun removeAuthStateListener(listener: AuthStateListener) {
        authListeners.remove(listener)
    }

    private fun notifyListeners() {
        authListeners.forEach { it.onAuthStateChanged(isAuthenticated, currentUser) }
    }

    /**
     * Check if user is already authenticated
     */
    fun checkAuthenticationStatus() {
        // Bypass authentication for development/testing
        if (BYPASS_AUTH) {
            currentUser = User(
                id = "dev_user",
                name = "Developer",
                phoneNumber = "+1234567890"
            )
            isAuthenticated = true
            Log.d(TAG, "Auth bypassed - using dev user")
            notifyListeners()
            return
        }

        val firebaseUser = auth.currentUser
        if (firebaseUser != null) {
            // User is signed in with Firebase
            val savedName = prefs.getString(KEY_USER_NAME, "") ?: ""
            val savedPhone = prefs.getString(KEY_USER_PHONE, "") ?: ""

            currentUser = User(
                id = firebaseUser.uid,
                name = savedName,
                phoneNumber = savedPhone.ifEmpty { firebaseUser.phoneNumber ?: "" }
            )
            isAuthenticated = true
            Log.d(TAG, "User authenticated from Firebase: ${currentUser?.phoneNumber}")
        } else if (prefs.getBoolean(KEY_IS_AUTHENTICATED, false)) {
            // Check local storage for demo/migration
            val userId = prefs.getString(KEY_USER_ID, null)
            val userName = prefs.getString(KEY_USER_NAME, null)
            val userPhone = prefs.getString(KEY_USER_PHONE, null)

            if (userId != null && userName != null && userPhone != null) {
                currentUser = User(id = userId, name = userName, phoneNumber = userPhone)
                isAuthenticated = true
                Log.d(TAG, "User authenticated from local storage: $userPhone")
            }
        } else {
            isAuthenticated = false
            currentUser = null
            Log.d(TAG, "User not authenticated")
        }
        notifyListeners()
    }

    /**
     * Check if user is locked out from OTP attempts
     */
    fun isLockedOut(): Boolean {
        val lockoutTime = prefs.getLong(KEY_LOCKOUT_TIME, 0)
        if (lockoutTime > 0) {
            val elapsed = System.currentTimeMillis() - lockoutTime
            if (elapsed < LOCKOUT_DURATION_MS) {
                return true
            } else {
                // Lockout expired, reset
                prefs.edit()
                    .remove(KEY_LOCKOUT_TIME)
                    .putInt(KEY_OTP_ATTEMPTS, 0)
                    .apply()
            }
        }
        return false
    }

    /**
     * Get remaining lockout time in seconds
     */
    fun getLockoutRemainingSeconds(): Int {
        val lockoutTime = prefs.getLong(KEY_LOCKOUT_TIME, 0)
        if (lockoutTime > 0) {
            val elapsed = System.currentTimeMillis() - lockoutTime
            val remaining = LOCKOUT_DURATION_MS - elapsed
            if (remaining > 0) {
                return (remaining / 1000).toInt()
            }
        }
        return 0
    }

    /**
     * Check if rate limited for OTP requests
     */
    fun isRateLimited(): Boolean {
        val lastRequestTime = prefs.getLong(KEY_LAST_OTP_REQUEST_TIME, 0)
        val requestCount = prefs.getInt(KEY_OTP_REQUESTS, 0)

        if (System.currentTimeMillis() - lastRequestTime > HOUR_MS) {
            // Reset counter after an hour
            prefs.edit().putInt(KEY_OTP_REQUESTS, 0).apply()
            return false
        }

        return requestCount >= MAX_OTP_REQUESTS_PER_HOUR
    }

    /**
     * Send OTP to phone number
     */
    fun sendOTP(
        phoneNumber: String,
        activity: Activity,
        onCodeSent: () -> Unit,
        onError: (String) -> Unit
    ) {
        if (isLockedOut()) {
            val remaining = getLockoutRemainingSeconds() / 60
            onError("Too many attempts. Try again in $remaining minutes.")
            return
        }

        if (isRateLimited()) {
            onError("Too many OTP requests. Please try again later.")
            return
        }

        isLoading = true
        errorMessage = null

        // Track OTP request for rate limiting
        val requestCount = prefs.getInt(KEY_OTP_REQUESTS, 0)
        prefs.edit()
            .putInt(KEY_OTP_REQUESTS, requestCount + 1)
            .putLong(KEY_LAST_OTP_REQUEST_TIME, System.currentTimeMillis())
            .apply()

        Log.d(TAG, "Sending OTP to: $phoneNumber")

        val callbacks = object : PhoneAuthProvider.OnVerificationStateChangedCallbacks() {
            override fun onVerificationCompleted(credential: PhoneAuthCredential) {
                // Auto-verification (rare on most devices)
                Log.d(TAG, "Auto-verification completed")
                isLoading = false
            }

            override fun onVerificationFailed(e: FirebaseException) {
                Log.e(TAG, "Verification failed", e)
                isLoading = false
                errorMessage = when {
                    e.message?.contains("blocked") == true ->
                        "Too many requests. Please try again later."
                    e.message?.contains("invalid") == true ->
                        "Invalid phone number format."
                    e.message?.contains("network") == true ->
                        "Network error. Please check your connection."
                    else -> e.message ?: "Verification failed"
                }
                onError(errorMessage!!)
            }

            override fun onCodeSent(
                verificationId: String,
                token: PhoneAuthProvider.ForceResendingToken
            ) {
                Log.d(TAG, "OTP code sent successfully")
                this@AuthenticationManager.verificationId = verificationId
                this@AuthenticationManager.resendToken = token
                isLoading = false
                onCodeSent()
            }
        }

        val options = PhoneAuthOptions.newBuilder(auth)
            .setPhoneNumber(phoneNumber)
            .setTimeout(60L, TimeUnit.SECONDS)
            .setActivity(activity)
            .setCallbacks(callbacks)
            .build()

        PhoneAuthProvider.verifyPhoneNumber(options)
    }

    /**
     * Resend OTP code
     */
    fun resendOTP(
        phoneNumber: String,
        activity: Activity,
        onCodeSent: () -> Unit,
        onError: (String) -> Unit
    ) {
        val token = resendToken
        if (token == null) {
            sendOTP(phoneNumber, activity, onCodeSent, onError)
            return
        }

        if (isRateLimited()) {
            onError("Too many OTP requests. Please try again later.")
            return
        }

        isLoading = true

        val callbacks = object : PhoneAuthProvider.OnVerificationStateChangedCallbacks() {
            override fun onVerificationCompleted(credential: PhoneAuthCredential) {
                isLoading = false
            }

            override fun onVerificationFailed(e: FirebaseException) {
                Log.e(TAG, "Resend verification failed", e)
                isLoading = false
                onError(e.message ?: "Failed to resend code")
            }

            override fun onCodeSent(
                verificationId: String,
                newToken: PhoneAuthProvider.ForceResendingToken
            ) {
                this@AuthenticationManager.verificationId = verificationId
                this@AuthenticationManager.resendToken = newToken
                isLoading = false
                onCodeSent()
            }
        }

        val options = PhoneAuthOptions.newBuilder(auth)
            .setPhoneNumber(phoneNumber)
            .setTimeout(60L, TimeUnit.SECONDS)
            .setActivity(activity)
            .setCallbacks(callbacks)
            .setForceResendingToken(token)
            .build()

        PhoneAuthProvider.verifyPhoneNumber(options)
    }

    /**
     * Verify OTP code and sign in
     */
    fun verifyOTP(
        otp: String,
        name: String,
        phoneNumber: String,
        onSuccess: () -> Unit,
        onError: (String) -> Unit
    ) {
        val verId = verificationId
        if (verId == null) {
            onError("Verification session expired. Please request a new code.")
            return
        }

        if (isLockedOut()) {
            val remaining = getLockoutRemainingSeconds() / 60
            onError("Account locked. Try again in $remaining minutes.")
            return
        }

        isLoading = true
        errorMessage = null

        val credential = PhoneAuthProvider.getCredential(verId, otp)

        auth.signInWithCredential(credential)
            .addOnCompleteListener { task ->
                isLoading = false

                if (task.isSuccessful) {
                    val user = task.result?.user
                    if (user != null) {
                        // Reset attempt counter on success
                        prefs.edit().putInt(KEY_OTP_ATTEMPTS, 0).apply()

                        // Save user data
                        currentUser = User(
                            id = user.uid,
                            name = name,
                            phoneNumber = phoneNumber
                        )
                        isAuthenticated = true

                        prefs.edit()
                            .putBoolean(KEY_IS_AUTHENTICATED, true)
                            .putString(KEY_USER_ID, user.uid)
                            .putString(KEY_USER_NAME, name)
                            .putString(KEY_USER_PHONE, phoneNumber)
                            .apply()

                        Log.d(TAG, "User signed in successfully: $phoneNumber")
                        notifyListeners()
                        onSuccess()
                    } else {
                        onError("Sign in failed. Please try again.")
                    }
                } else {
                    // Increment failed attempt counter
                    val attempts = prefs.getInt(KEY_OTP_ATTEMPTS, 0) + 1
                    prefs.edit().putInt(KEY_OTP_ATTEMPTS, attempts).apply()

                    if (attempts >= MAX_OTP_ATTEMPTS) {
                        // Lock out user
                        prefs.edit().putLong(KEY_LOCKOUT_TIME, System.currentTimeMillis()).apply()
                        onError("Too many failed attempts. Account locked for 30 minutes.")
                    } else {
                        val remaining = MAX_OTP_ATTEMPTS - attempts
                        errorMessage = "Invalid code. $remaining attempts remaining."
                        onError(errorMessage!!)
                    }
                    Log.e(TAG, "Sign in failed", task.exception)
                }
            }
    }

    /**
     * Sign out user
     */
    fun logout() {
        auth.signOut()

        prefs.edit()
            .remove(KEY_IS_AUTHENTICATED)
            .remove(KEY_USER_ID)
            .remove(KEY_USER_NAME)
            .remove(KEY_USER_PHONE)
            .apply()

        isAuthenticated = false
        currentUser = null
        verificationId = null
        resendToken = null

        Log.d(TAG, "User signed out")
        notifyListeners()
    }

    /**
     * Get current user's display name
     */
    fun getUserName(): String {
        return currentUser?.name ?: prefs.getString(KEY_USER_NAME, "") ?: ""
    }

    /**
     * Get current user's phone number
     */
    fun getUserPhone(): String {
        return currentUser?.phoneNumber ?: prefs.getString(KEY_USER_PHONE, "") ?: ""
    }
}
