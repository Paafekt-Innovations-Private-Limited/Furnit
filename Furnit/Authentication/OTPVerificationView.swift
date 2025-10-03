import SwiftUI

struct OTPVerificationView: View {
    let name: String
    let phoneNumber: String
    let expectedOTP: String
    let authManager: AuthenticationManager
    
    @State private var otpDigits: [String] = Array(repeating: "", count: 6)
    @State private var showError = false
    @State private var errorMessage = ""
    @State private var isVerifying = false
    @State private var navigateToHome = false
    @State private var resendTimer = 30
    @State private var canResend = false
    @FocusState private var focusedField: Int?
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        ZStack {
            // Background gradient
            LinearGradient(
                colors: [Color.blue.opacity(0.8), Color.purple.opacity(0.6)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
            
            VStack(spacing: 30) {
                // Back Button
                HStack {
                    Button(action: { dismiss() }) {
                        HStack(spacing: 4) {
                            Image(systemName: "chevron.left")
                            Text("Back")
                        }
                        .foregroundColor(.white)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(Color.white.opacity(0.2))
                        .cornerRadius(20)
                    }
                    Spacer()
                }
                .padding()
                
                Spacer()
                
                // Header
                VStack(spacing: 16) {
                    Image(systemName: "lock.shield.fill")
                        .font(.system(size: 50))
                        .foregroundColor(.white)
                        .shadow(radius: 10)
                    
                    Text("Verify OTP")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                    
                    Text("Enter the code sent to")
                        .font(.subheadline)
                        .foregroundColor(.white.opacity(0.9))
                    
                    Text(phoneNumber)
                        .font(.headline)
                        .foregroundColor(.white)
                }
                
                // OTP Input Fields
                VStack(spacing: 20) {
                    HStack(spacing: 12) {
                        ForEach(0..<6, id: \.self) { index in
                            OTPDigitField(
                                digit: $otpDigits[index],
                                isActive: focusedField == index
                            )
                            .focused($focusedField, equals: index)
                            .onChange(of: otpDigits[index]) { oldValue, newValue in
                                handleOTPInput(at: index, oldValue: oldValue, newValue: newValue)
                            }
                        }
                    }
                    
                    // Demo hint
                    Text("Demo Mode: Enter 123456")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.7))
                    
                    // Verify Button
                    Button(action: verifyOTP) {
                        HStack {
                            if isVerifying {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(0.8)
                            } else {
                                Image(systemName: "checkmark.shield.fill")
                            }
                            Text(isVerifying ? "Verifying..." : "Verify")
                                .fontWeight(.semibold)
                        }
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(isOTPComplete ? Color.green : Color.gray)
                        )
                    }
                    .disabled(!isOTPComplete || isVerifying)
                    
                    // Resend OTP
                    HStack {
                        if canResend {
                            Button(action: resendOTP) {
                                HStack {
                                    Image(systemName: "arrow.clockwise")
                                    Text("Resend OTP")
                                }
                                .foregroundColor(.white)
                            }
                        } else {
                            Text("Resend in \(resendTimer)s")
                                .foregroundColor(.white.opacity(0.7))
                                .font(.caption)
                        }
                    }
                }
                .padding(24)
                .background(
                    RoundedRectangle(cornerRadius: 20)
                        .fill(Color.white.opacity(0.15))
                        .background(
                            RoundedRectangle(cornerRadius: 20)
                                .stroke(Color.white.opacity(0.3), lineWidth: 1)
                        )
                )
                .padding(.horizontal)
                
                Spacer()
                Spacer()
            }
        }
        .navigationBarHidden(true)
        .onAppear {
            focusedField = 0
            startResendTimer()
        }
        .alert("Error", isPresented: $showError) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(errorMessage)
        }
        .fullScreenCover(isPresented: $navigateToHome) {
            ContentView() // This will show HomeView since user is authenticated
        }
    }
    
    private var isOTPComplete: Bool {
        otpDigits.allSatisfy { !$0.isEmpty }
    }
    
    private var enteredOTP: String {
        otpDigits.joined()
    }
    
    private func handleOTPInput(at index: Int, oldValue: String, newValue: String) {
        // Ensure only one digit
        if newValue.count > 1 {
            otpDigits[index] = String(newValue.suffix(1))
            return
        }
        
        // Move to next field if digit entered
        if !newValue.isEmpty && index < 5 {
            focusedField = index + 1
        }
        
        // Handle paste
        if newValue.count == 0 && oldValue.count > 0 {
            // Backspace pressed
            if index > 0 {
                focusedField = index - 1
            }
        }
        
        // Auto-verify when complete
        if isOTPComplete {
            verifyOTP()
        }
    }
    
    private func verifyOTP() {
        isVerifying = true
        
        // Simulate API delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            if authManager.verifyOTP(enteredOTP, actualOTP: expectedOTP) {
                // Success - login the user
                authManager.login(name: name, phoneNumber: phoneNumber)
                
                // Haptic feedback
                let impactFeedback = UIImpactFeedbackGenerator(style: .medium)
                impactFeedback.impactOccurred()
                
                // Navigate to home
                navigateToHome = true
            } else {
                // Error - wrong OTP
                errorMessage = "Invalid OTP. Please try again."
                showError = true
                isVerifying = false
                
                // Clear fields
                otpDigits = Array(repeating: "", count: 6)
                focusedField = 0
            }
        }
    }
    
    private func resendOTP() {
        // Reset timer
        canResend = false
        resendTimer = 30
        startResendTimer()
        
        // Request new OTP
        _ = authManager.sendOTP(to: phoneNumber)
        
        // Show confirmation
        let impactFeedback = UIImpactFeedbackGenerator(style: .light)
        impactFeedback.impactOccurred()
    }
    
    private func startResendTimer() {
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { timer in
            if resendTimer > 0 {
                resendTimer -= 1
            } else {
                canResend = true
                timer.invalidate()
            }
        }
    }
}

// OTP Digit Field Component
struct OTPDigitField: View {
    @Binding var digit: String
    let isActive: Bool
    
    var body: some View {
        TextField("", text: $digit)
            .keyboardType(.numberPad)
            .multilineTextAlignment(.center)
            .frame(width: 45, height: 55)
            .background(Color.white.opacity(0.95))
            .cornerRadius(10)
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(isActive ? Color.blue : Color.clear, lineWidth: 2)
            )
            .foregroundColor(.black)
            .font(.title2.bold())
            .onChange(of: digit) { _, newValue in
                // Only allow digits
                digit = newValue.filter { $0.isNumber }
            }
    }
}

#Preview {
    OTPVerificationView(
        name: "John Doe",
        phoneNumber: "123-456-7890",
        expectedOTP: "123456",
        authManager: AuthenticationManager()
    )
}
