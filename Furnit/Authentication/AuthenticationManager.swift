import Foundation
import SwiftUI

class AuthenticationManager: ObservableObject {
    @Published var isAuthenticated = false
    @Published var currentUser: User?
    
    // Store in UserDefaults for persistence
    private let userDefaults = UserDefaults.standard
    private let authKey = "isAuthenticated"
    private let userNameKey = "userName"
    private let userPhoneKey = "userPhone"
    
    struct User {
        let name: String
        let phoneNumber: String
    }
    
    init() {
        // Check if user was previously logged in
        checkAuthenticationStatus()
    }
    
    private func checkAuthenticationStatus() {
        isAuthenticated = userDefaults.bool(forKey: authKey)
        
        if isAuthenticated,
           let name = userDefaults.string(forKey: userNameKey),
           let phone = userDefaults.string(forKey: userPhoneKey) {
            currentUser = User(name: name, phoneNumber: phone)
        }
    }
    
    func sendOTP(to phoneNumber: String) -> String {
        // In production, this would send a real OTP via SMS
        // For demo, generate a random 6-digit OTP
        let otp = String(format: "%06d", Int.random(in: 0...999999))
        
        // For testing, print the OTP to console
        print("📱 OTP sent to \(phoneNumber): \(otp)")
        
        // In demo mode, we'll use a fixed OTP for testing
        return "123456"
    }
    
    func verifyOTP(_ inputOTP: String, actualOTP: String) -> Bool {
        // In production, verify with backend
        // For demo, check against the generated OTP or accept "123456"
        return inputOTP == actualOTP || inputOTP == "123456"
    }
    
    func login(name: String, phoneNumber: String) {
        // Save user data
        userDefaults.set(true, forKey: authKey)
        userDefaults.set(name, forKey: userNameKey)
        userDefaults.set(phoneNumber, forKey: userPhoneKey)
        
        currentUser = User(name: name, phoneNumber: phoneNumber)
        isAuthenticated = true
        
        print("✅ User logged in: \(name) - \(phoneNumber)")
    }
    
    func logout() {
        // Clear user data
        userDefaults.set(false, forKey: authKey)
        userDefaults.removeObject(forKey: userNameKey)
        userDefaults.removeObject(forKey: userPhoneKey)
        
        currentUser = nil
        isAuthenticated = false
        
        print("👋 User logged out")
    }
    
    func formatPhoneNumber(_ number: String) -> String {
        // Remove non-numeric characters
        let cleanNumber = number.filter { $0.isNumber }
        
        // Format as needed (example: XXX-XXX-XXXX for US numbers)
        if cleanNumber.count == 10 {
            let areaCode = String(cleanNumber.prefix(3))
            let middle = String(cleanNumber.dropFirst(3).prefix(3))
            let last = String(cleanNumber.dropFirst(6))
            return "\(areaCode)-\(middle)-\(last)"
        }
        
        return cleanNumber
    }
}
