import SwiftUI

struct OTPVerificationView: View {
    let name: String
    let phoneNumber: String
    let authManager: AuthenticationManager

    @State private var otpDigits: [String] = Array(repeating: "", count: 6)
    @State private var showError = false
    @State private var errorMessage = ""
    @State private var isVerifying = false
    @State private var navigateToHome = false
    @State private var resendTimer = 30
    @State private var canResend = false
    @State private var isResending = false
    @FocusState private var focusedField: Int?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack {
            Theme.Palette.background
                .ignoresSafeArea()

            VStack(spacing: 30) {
                Spacer()
                    .frame(height: 60)

                // Header
                VStack(spacing: 16) {
                    Image(systemName: "lock.shield.fill")
                        .font(.system(size: 50))
                        .foregroundStyle(Theme.Palette.accent)
                        .shadow(color: Theme.Palette.accent.opacity(0.25), radius: 10)

                    Text(L10n.OTP.title)
                        .font(Theme.Typo.display())
                        .foregroundStyle(Theme.Palette.textPrimary)

                    Text(L10n.OTP.subtitle)
                        .font(Theme.Typo.body())
                        .foregroundStyle(Theme.Palette.textSecondary)

                    Text(phoneNumber)
                        .font(Theme.Typo.headline())
                        .foregroundStyle(Theme.Palette.textPrimary)
                }

                // OTP Input Fields
                VStack(spacing: 20) {
                    HStack(spacing: 12) {
                        ForEach(0..<6, id: \.self) { index in
                            OTPDigitField(
                                digit: $otpDigits[index],
                                isActive: focusedField == index,
                                isFirstField: index == 0,
                                onPaste: { pastedCode in
                                    handlePastedOTP(pastedCode)
                                }
                            )
                            .focused($focusedField, equals: index)
                            .onChange(of: otpDigits[index]) { oldValue, newValue in
                                handleOTPInput(at: index, oldValue: oldValue, newValue: newValue)
                            }
                            .disabled(isVerifying)
                        }
                    }

                    // Verify Button
                    Button(action: verifyOTP) {
                        HStack {
                            if isVerifying {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: Theme.Palette.accentText))
                                    .scaleEffect(0.8)
                            } else {
                                Image(systemName: "checkmark.shield.fill")
                            }
                            Text(isVerifying ? L10n.OTP.verifying : L10n.OTP.verify)
                        }
                    }
                    .buttonStyle(PrimaryButtonStyle())
                    .disabled(!isOTPComplete || isVerifying)
                    .opacity(isOTPComplete ? 1.0 : 0.7)

                    // Resend OTP
                    HStack {
                        if canResend {
                            Button(action: resendOTP) {
                                HStack {
                                    if isResending {
                                        ProgressView()
                                            .progressViewStyle(CircularProgressViewStyle(tint: Theme.Palette.accent))
                                            .scaleEffect(0.6)
                                    } else {
                                        Image(systemName: "arrow.clockwise")
                                    }
                                    Text(isResending ? L10n.Login.sending : L10n.OTP.resend)
                                }
                                .foregroundStyle(Theme.Palette.accent)
                            }
                            .disabled(isResending)
                        } else {
                            Text(L10n.OTP.resendIn(resendTimer))
                                .foregroundStyle(Theme.Palette.textSecondary)
                                .font(Theme.Typo.caption())
                        }
                    }
                }
                .padding(Theme.Space.xl)
                .paafektCardSurface()
                .padding(.horizontal)

                Spacer()
                Spacer()
            }
        }
        .overlay(alignment: .topLeading) {
            // Back Button - placed on top layer
            Button(action: { dismiss() }) {
                HStack(spacing: 4) {
                    Image(systemName: "chevron.left")
                    Text(L10n.Common.back)
                }
                .foregroundStyle(Theme.Palette.textPrimary)
                .padding(.horizontal, Theme.Space.lg)
                .padding(.vertical, Theme.Space.sm)
                .background(Theme.Palette.surfaceHi)
                .cornerRadius(Theme.Radius.sheet)
            }
            .padding(.top, 60)
            .padding(.leading, 16)
        }
        .navigationBarHidden(true)
        .onAppear {
            focusedField = 0
            startResendTimer()
        }
        .alert(L10n.Common.error, isPresented: $showError) {
            Button(L10n.Common.ok, role: .cancel) { }
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

    private func handlePastedOTP(_ code: String) {
        // Distribute pasted OTP across all fields
        let digits = Array(code.prefix(6))
        for (index, char) in digits.enumerated() {
            otpDigits[index] = String(char)
        }
        // Move focus to last filled field or end
        focusedField = min(digits.count, 5)

        // Auto-verify if complete
        if digits.count >= 6 && !isVerifying {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                verifyOTP()
            }
        }
    }

    private func handleOTPInput(at index: Int, oldValue: String, newValue: String) {
        // Skip if this was handled by paste
        if newValue.count > 1 {
            return
        }

        // Move to next field if digit entered
        if !newValue.isEmpty && index < 5 {
            focusedField = index + 1
        }

        // Handle backspace
        if newValue.count == 0 && oldValue.count > 0 {
            if index > 0 {
                focusedField = index - 1
            }
        }

        // Auto-verify when complete
        if isOTPComplete && !isVerifying {
            verifyOTP()
        }
    }

    private func verifyOTP() {
        guard !isVerifying else { return }
        isVerifying = true

        authManager.verifyOTP(enteredOTP, name: name, phoneNumber: phoneNumber) { success, error in
            if success {
                // Haptic feedback
                let impactFeedback = UIImpactFeedbackGenerator(style: .medium)
                impactFeedback.impactOccurred()

                // Navigate to home
                navigateToHome = true
            } else {
                // Error - wrong OTP
                errorMessage = error ?? L10n.OTP.invalidError
                showError = true
                isVerifying = false

                // Clear fields
                otpDigits = Array(repeating: "", count: 6)
                focusedField = 0
            }
        }
    }

    private func resendOTP() {
        isResending = true

        authManager.sendOTP(to: phoneNumber) { success, error in
            isResending = false

            if success {
                // Reset timer
                canResend = false
                resendTimer = 30
                startResendTimer()

                // Haptic feedback
                let impactFeedback = UIImpactFeedbackGenerator(style: .light)
                impactFeedback.impactOccurred()
            } else {
                errorMessage = error ?? L10n.OTP.resendFailed
                showError = true
            }
        }
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
    let isFirstField: Bool
    var onPaste: ((String) -> Void)?

    init(digit: Binding<String>, isActive: Bool, isFirstField: Bool = false, onPaste: ((String) -> Void)? = nil) {
        self._digit = digit
        self.isActive = isActive
        self.isFirstField = isFirstField
        self.onPaste = onPaste
    }

    var body: some View {
        TextField("", text: $digit)
            .keyboardType(.numberPad)
            .textContentType(isFirstField ? .oneTimeCode : .none) // Enable SMS autofill on first field
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
                // Handle paste of full OTP code
                let filtered = newValue.filter { $0.isNumber }
                if filtered.count > 1 {
                    onPaste?(filtered)
                    return
                }
                digit = filtered
            }
    }
}

#Preview {
    OTPVerificationView(
        name: "John Doe",
        phoneNumber: "123-456-7890",
        authManager: AuthenticationManager()
    )
}
