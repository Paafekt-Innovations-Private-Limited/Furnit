import Foundation
import SwiftUI
import FirebaseCore
import FirebaseAuth

/// Authentication Manager for Phone + OTP authentication
/// Production mode with Firebase Authentication
class AuthenticationManager: ObservableObject {
    @Published var isAuthenticated = false
    @Published var currentUser: User?
    @Published var isLoading = false
    @Published var errorMessage: String?

    // Firebase verification ID (used for OTP verification)
    private var verificationID: String?

    // Store in UserDefaults for persistence
    private let userDefaults = UserDefaults.standard
    private let authKey = "isAuthenticated"
    private let userNameKey = "userName"
    private let userPhoneKey = "userPhone"
    private let userIdKey = "userId"

    // Anti-bot protection
    private let otpAttemptsKey = "otpAttempts"
    private let otpLockoutKey = "otpLockoutUntil"
    private let otpRequestCountKey = "otpRequestCount"
    private let otpRequestWindowKey = "otpRequestWindow"
    private let maxOTPAttempts = 5
    private let maxOTPRequestsPerHour = 5
    private let lockoutDurationMinutes = 30.0

    struct User: Codable {
        let id: String
        let name: String
        let phoneNumber: String
    }

    init() {
        // Delay auth check to ensure Firebase is configured
        DispatchQueue.main.async { [weak self] in
            self?.checkAuthenticationStatus()
        }
    }

    // MARK: - Check Existing Auth

    private func checkAuthenticationStatus() {
        // Guard: Make sure Firebase is configured
        guard FirebaseApp.app() != nil else {
            AppLogger.authError("Firebase not configured yet, retrying...")
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
                self?.checkAuthenticationStatus()
            }
            return
        }

        // Check Firebase auth state first
        if let firebaseUser = Auth.auth().currentUser {
            // User is signed in with Firebase
            if let name = userDefaults.string(forKey: userNameKey),
               let phone = userDefaults.string(forKey: userPhoneKey) {
                currentUser = User(id: firebaseUser.uid, name: name, phoneNumber: phone)
                isAuthenticated = true
                return
            }
        }

        // Fallback to UserDefaults (for migration or demo)
        if userDefaults.bool(forKey: authKey),
           let name = userDefaults.string(forKey: userNameKey),
           let phone = userDefaults.string(forKey: userPhoneKey) {
            let userId = userDefaults.string(forKey: userIdKey) ?? UUID().uuidString
            currentUser = User(id: userId, name: name, phoneNumber: phone)
            isAuthenticated = true
        }
    }

    // MARK: - Send OTP

    func sendOTP(to phoneNumber: String, completion: @escaping (Bool, String?) -> Void) {
        isLoading = true
        errorMessage = nil

        // Guard: Make sure Firebase is configured
        guard FirebaseApp.app() != nil else {
            isLoading = false
            let error = "App is still initializing. Please try again."
            errorMessage = error
            AppLogger.authError("sendOTP called before Firebase configured")
            completion(false, error)
            return
        }

        // Anti-bot: Check rate limits
        let rateCheck = canRequestOTP()
        if !rateCheck.allowed {
            isLoading = false
            errorMessage = rateCheck.message
            completion(false, rateCheck.message)
            return
        }

        // Record this OTP request
        recordOTPRequest()

        // Format phone number with country code
        let formattedPhone = formatPhoneForFirebase(phoneNumber)

        // Debug logging
        if AppStateManager.shared.qualitySettings.debugMode {
            print("🔐 [Auth] Starting phone verification for: \(formattedPhone)")
            print("🔐 [Auth] Firebase App: \(FirebaseApp.app()?.name ?? "nil")")
            print("🔐 [Auth] Auth instance: \(Auth.auth())")
            print("🔐 [Auth] Auth settings: \(String(describing: Auth.auth().settings))")
            print("🔐 [Auth] Auth currentUser: \(String(describing: Auth.auth().currentUser))")
        }

        let provider = PhoneAuthProvider.provider()
        if AppStateManager.shared.qualitySettings.debugMode {
            print("🔐 [Auth] PhoneAuthProvider: \(provider)")
        }

        // Firebase Phone Auth
        provider.verifyPhoneNumber(formattedPhone, uiDelegate: nil) { [weak self] verificationID, error in
            DispatchQueue.main.async {
                self?.isLoading = false

                if let error = error {
                    let nsError = error as NSError
                    
                    // ✅ Detailed error logging - ONLY when debug mode is ON
                    if AppStateManager.shared.qualitySettings.debugMode {
                        print("🔐 [Auth] ❌ ERROR DETAILS:")
                        print("🔐 [Auth]   Code: \(nsError.code)")
                        print("🔐 [Auth]   Domain: \(nsError.domain)")
                        print("🔐 [Auth]   Description: \(nsError.localizedDescription)")
                        print("🔐 [Auth]   UserInfo: \(nsError.userInfo)")
                        print("🔐 [Auth]   Underlying error: \(String(describing: nsError.userInfo[NSUnderlyingErrorKey]))")
                        
                        // ✅ Additional Firebase App Check specific logging
                        if let underlyingError = nsError.userInfo[NSUnderlyingErrorKey] as? NSError {
                            print("🔐 [Auth]   --- Underlying Error Details ---")
                            print("🔐 [Auth]   Underlying Code: \(underlyingError.code)")
                            print("🔐 [Auth]   Underlying Domain: \(underlyingError.domain)")
                            print("🔐 [Auth]   Underlying UserInfo: \(underlyingError.userInfo)")
                            
                            if let failureReason = underlyingError.userInfo[NSLocalizedFailureReasonErrorKey] as? String {
                                print("🔐 [Auth]   Failure Reason: \(failureReason)")
                            }
                            
                            if let responseBody = underlyingError.userInfo["FIRAppCheckErrorResponseBodyKey"] as? String {
                                print("🔐 [Auth]   Response Body: \(responseBody)")
                            }
                        }
                    }
                    
                    // Always log basic error info (not debug-mode specific)
                    AppLogger.authError("OTP Error - Code: \(nsError.code), Domain: \(nsError.domain), Description: \(nsError.localizedDescription)")

                    // Show user-friendly message
                    var userMessage = error.localizedDescription
                    if nsError.code == 17999 {
                        userMessage = "Phone authentication failed. Please check your internet connection and try again."
                    } else if nsError.code == 17010 {
                        userMessage = "Too many requests. Please wait a few minutes and try again."
                    }

                    self?.errorMessage = userMessage
                    completion(false, userMessage)
                    return
                }

                self?.verificationID = verificationID
                AppLogger.authDebug("OTP sent to \(formattedPhone)")
                completion(true, nil)
            }
        }
    }

    // MARK: - Verify OTP

    func verifyOTP(_ otp: String, name: String, phoneNumber: String, completion: @escaping (Bool, String?) -> Void) {
        isLoading = true
        errorMessage = nil

        // Anti-bot: Check if locked out
        if isLockedOut() {
            isLoading = false
            let error = "Too many failed attempts. Try again in \(getLockoutRemainingTime())."
            errorMessage = error
            completion(false, error)
            return
        }

        guard let verificationID = verificationID else {
            isLoading = false
            let error = "Verification expired. Please request a new OTP."
            errorMessage = error
            completion(false, error)
            return
        }

        let credential = PhoneAuthProvider.provider().credential(
            withVerificationID: verificationID,
            verificationCode: otp
        )

        Auth.auth().signIn(with: credential) { [weak self] authResult, error in
            DispatchQueue.main.async {
                self?.isLoading = false

                if let error = error {
                    // Anti-bot: Record failed attempt
                    self?.recordFailedOTPAttempt()

                    let attempts = self?.userDefaults.integer(forKey: self?.otpAttemptsKey ?? "") ?? 0
                    let remaining = (self?.maxOTPAttempts ?? 5) - attempts
                    var errorMsg = error.localizedDescription
                    if remaining > 0 && remaining < 3 {
                        errorMsg += " (\(remaining) attempts remaining)"
                    }

                    self?.errorMessage = errorMsg
                    AppLogger.authError("OTP verify error: \(error.localizedDescription)")
                    completion(false, errorMsg)
                    return
                }

                if let user = authResult?.user {
                    // Anti-bot: Reset attempts on successful login
                    self?.resetOTPAttempts()

                    self?.loginUser(name: name, phoneNumber: phoneNumber, userId: user.uid)
                    AppLogger.authDebug("Firebase auth successful: \(user.uid)")
                    completion(true, nil)
                }
            }
        }
    }

    // MARK: - Login/Logout

    private func loginUser(name: String, phoneNumber: String, userId: String) {
        // Save to UserDefaults
        userDefaults.set(true, forKey: authKey)
        userDefaults.set(name, forKey: userNameKey)
        userDefaults.set(phoneNumber, forKey: userPhoneKey)
        userDefaults.set(userId, forKey: userIdKey)

        // Update state
        currentUser = User(id: userId, name: name, phoneNumber: phoneNumber)
        isAuthenticated = true

        AppLogger.authDebug("User logged in: \(name)")
    }

    func logout() {
        // Clear UserDefaults
        userDefaults.set(false, forKey: authKey)
        userDefaults.removeObject(forKey: userNameKey)
        userDefaults.removeObject(forKey: userPhoneKey)
        userDefaults.removeObject(forKey: userIdKey)

        // Sign out from Firebase
        do {
            try Auth.auth().signOut()
            AppLogger.authDebug("Firebase sign out successful")
        } catch {
            AppLogger.authError("Firebase sign out error: \(error.localizedDescription)")
        }

        // Update state
        currentUser = nil
        isAuthenticated = false
        verificationID = nil

        AppLogger.authDebug("User logged out")
    }

    // MARK: - Anti-Bot Protection

    private func isLockedOut() -> Bool {
        if let lockoutDate = userDefaults.object(forKey: otpLockoutKey) as? Date {
            if Date() < lockoutDate {
                return true
            } else {
                // Lockout expired, reset
                userDefaults.removeObject(forKey: otpLockoutKey)
                userDefaults.set(0, forKey: otpAttemptsKey)
            }
        }
        return false
    }

    private func getLockoutRemainingTime() -> String {
        if let lockoutDate = userDefaults.object(forKey: otpLockoutKey) as? Date {
            let remaining = lockoutDate.timeIntervalSince(Date())
            if remaining > 0 {
                let minutes = Int(remaining / 60)
                return "\(minutes) minute\(minutes == 1 ? "" : "s")"
            }
        }
        return "0 minutes"
    }

    private func recordFailedOTPAttempt() {
        let attempts = userDefaults.integer(forKey: otpAttemptsKey) + 1
        userDefaults.set(attempts, forKey: otpAttemptsKey)

        if attempts >= maxOTPAttempts {
            let lockoutUntil = Date().addingTimeInterval(lockoutDurationMinutes * 60)
            userDefaults.set(lockoutUntil, forKey: otpLockoutKey)
            AppLogger.warning("Account locked for \(lockoutDurationMinutes) minutes due to too many failed attempts", category: AppLogger.auth)
        }
    }

    private func resetOTPAttempts() {
        userDefaults.set(0, forKey: otpAttemptsKey)
        userDefaults.removeObject(forKey: otpLockoutKey)
    }

    private func canRequestOTP() -> (allowed: Bool, message: String?) {
        // Check lockout first
        if isLockedOut() {
            return (false, "Too many failed attempts. Try again in \(getLockoutRemainingTime()).")
        }

        // Check hourly rate limit
        let now = Date()
        let windowStart = userDefaults.object(forKey: otpRequestWindowKey) as? Date ?? Date.distantPast
        let requestCount = userDefaults.integer(forKey: otpRequestCountKey)

        // Reset window if hour has passed
        if now.timeIntervalSince(windowStart) > 3600 {
            userDefaults.set(now, forKey: otpRequestWindowKey)
            userDefaults.set(0, forKey: otpRequestCountKey)
            return (true, nil)
        }

        if requestCount >= maxOTPRequestsPerHour {
            let remainingTime = Int((3600 - now.timeIntervalSince(windowStart)) / 60)
            return (false, "Too many OTP requests. Try again in \(remainingTime) minutes.")
        }

        return (true, nil)
    }

    private func recordOTPRequest() {
        let now = Date()
        let windowStart = userDefaults.object(forKey: otpRequestWindowKey) as? Date ?? now

        if now.timeIntervalSince(windowStart) > 3600 {
            userDefaults.set(now, forKey: otpRequestWindowKey)
            userDefaults.set(1, forKey: otpRequestCountKey)
        } else {
            let count = userDefaults.integer(forKey: otpRequestCountKey) + 1
            userDefaults.set(count, forKey: otpRequestCountKey)
        }
    }

    // MARK: - Helper Functions

    private func formatPhoneForFirebase(_ phone: String) -> String {
        // Phone number should already include country code from LoginView
        // Just ensure it starts with +
        let cleaned = phone.trimmingCharacters(in: .whitespaces)
        if cleaned.hasPrefix("+") {
            return cleaned
        }
        return "+\(cleaned)"
    }

    func formatPhoneNumber(_ number: String) -> String {
        let cleanNumber = number.filter { $0.isNumber }

        // Format as XXX-XXX-XXXX
        if cleanNumber.count == 10 {
            let first = String(cleanNumber.prefix(3))
            let middle = String(cleanNumber.dropFirst(3).prefix(3))
            let last = String(cleanNumber.dropFirst(6))
            return "\(first)-\(middle)-\(last)"
        }

        return cleanNumber
    }
}
