import Foundation
import SwiftUI
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

    struct User: Codable {
        let id: String
        let name: String
        let phoneNumber: String
    }

    init() {
        checkAuthenticationStatus()
    }

    // MARK: - Check Existing Auth

    private func checkAuthenticationStatus() {
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

        // Format phone number with country code
        let formattedPhone = formatPhoneForFirebase(phoneNumber)

        // Firebase Phone Auth
        PhoneAuthProvider.provider().verifyPhoneNumber(formattedPhone, uiDelegate: nil) { [weak self] verificationID, error in
            DispatchQueue.main.async {
                self?.isLoading = false

                if let error = error {
                    self?.errorMessage = error.localizedDescription
                    print("❌ OTP send error: \(error.localizedDescription)")
                    completion(false, error.localizedDescription)
                    return
                }

                self?.verificationID = verificationID
                print("📱 OTP sent to \(formattedPhone)")
                completion(true, nil)
            }
        }
    }

    // MARK: - Verify OTP

    func verifyOTP(_ otp: String, name: String, phoneNumber: String, completion: @escaping (Bool, String?) -> Void) {
        isLoading = true
        errorMessage = nil

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
                    self?.errorMessage = error.localizedDescription
                    print("❌ OTP verify error: \(error.localizedDescription)")
                    completion(false, error.localizedDescription)
                    return
                }

                if let user = authResult?.user {
                    self?.loginUser(name: name, phoneNumber: phoneNumber, userId: user.uid)
                    print("✅ Firebase auth successful: \(user.uid)")
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

        print("✅ User logged in: \(name) - \(phoneNumber)")
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
            print("✅ Firebase sign out successful")
        } catch {
            print("❌ Firebase sign out error: \(error.localizedDescription)")
        }

        // Update state
        currentUser = nil
        isAuthenticated = false
        verificationID = nil

        print("👋 User logged out")
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

    // MARK: - Legacy support (for existing views)

    func sendOTP(to phoneNumber: String) -> String {
        // Legacy method - triggers Firebase OTP
        sendOTP(to: phoneNumber) { _, _ in }
        return ""
    }

    func verifyOTP(_ inputOTP: String, actualOTP: String) -> Bool {
        // Legacy method - not used in Firebase flow
        return false
    }

    func login(name: String, phoneNumber: String) {
        // Legacy method for backward compatibility
        loginUser(name: name, phoneNumber: phoneNumber, userId: UUID().uuidString)
    }
}
