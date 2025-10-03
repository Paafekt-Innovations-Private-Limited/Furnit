import SwiftUI

struct LoginView: View {
    @StateObject private var authManager = AuthenticationManager()
    @State private var name = ""
    @State private var phoneNumber = ""
    @State private var showOTPView = false
    @State private var generatedOTP = ""
    @State private var showError = false
    @State private var errorMessage = ""
    @FocusState private var nameFieldFocused: Bool
    @FocusState private var phoneFieldFocused: Bool
    
    var body: some View {
        NavigationStack {
            ZStack {
                // Background gradient
                LinearGradient(
                    colors: [Color.blue.opacity(0.8), Color.purple.opacity(0.6)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()
                
                VStack(spacing: 30) {
                    Spacer()
                    
                    // App Logo/Icon
                    VStack(spacing: 16) {
                        Image(systemName: "cube.fill")
                            .font(.system(size: 60))
                            .foregroundColor(.white)
                            .shadow(radius: 10)
                        
                        Text("Furnit")
                            .font(.largeTitle)
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                        
                        Text("3D Room Explorer")
                            .font(.subheadline)
                            .foregroundColor(.white.opacity(0.9))
                    }
                    
                    // Login Form
                    VStack(spacing: 20) {
                        // Name Field
                        VStack(alignment: .leading, spacing: 8) {
                            Label("Your Name", systemImage: "person.fill")
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.9))
                            
                            TextField("Enter your name", text: $name)
                                .textFieldStyle(CustomTextFieldStyle())
                                .focused($nameFieldFocused)
                                .submitLabel(.next)
                                .onSubmit {
                                    phoneFieldFocused = true
                                }
                        }
                        
                        // Phone Field
                        VStack(alignment: .leading, spacing: 8) {
                            Label("Phone Number", systemImage: "phone.fill")
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.9))
                            
                            TextField("Enter phone number", text: $phoneNumber)
                                .textFieldStyle(CustomTextFieldStyle())
                                .keyboardType(.phonePad)
                                .focused($phoneFieldFocused)
                                .onChange(of: phoneNumber) { _, newValue in
                                    // Auto-format phone number
                                    phoneNumber = formatPhoneInput(newValue)
                                }
                        }
                        
                        // Send OTP Button
                        Button(action: sendOTP) {
                            HStack {
                                Image(systemName: "paperplane.fill")
                                Text("Send OTP")
                                    .fontWeight(.semibold)
                            }
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(isValidInput ? Color.blue : Color.gray)
                            )
                        }
                        .disabled(!isValidInput)
                        .scaleEffect(isValidInput ? 1.0 : 0.95)
                        .animation(.easeInOut(duration: 0.2), value: isValidInput)
                        
                        // Demo Mode Hint
                        Text("Demo Mode: Use OTP 123456")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.7))
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
            .navigationDestination(isPresented: $showOTPView) {
                OTPVerificationView(
                    name: name,
                    phoneNumber: phoneNumber,
                    expectedOTP: generatedOTP,
                    authManager: authManager
                )
            }
            .alert("Error", isPresented: $showError) {
                Button("OK", role: .cancel) { }
            } message: {
                Text(errorMessage)
            }
            .onTapGesture {
                // Dismiss keyboard when tapping outside
                nameFieldFocused = false
                phoneFieldFocused = false
            }
        }
    }
    
    private var isValidInput: Bool {
        !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        phoneNumber.filter { $0.isNumber }.count >= 10
    }
    
    private func sendOTP() {
        guard isValidInput else {
            errorMessage = "Please enter valid name and phone number"
            showError = true
            return
        }
        
        // Generate OTP
        generatedOTP = authManager.sendOTP(to: phoneNumber)
        
        // Navigate to OTP view
        showOTPView = true
    }
    
    private func formatPhoneInput(_ input: String) -> String {
        let cleaned = input.filter { $0.isNumber }
        let limited = String(cleaned.prefix(10)) // Limit to 10 digits
        
        if limited.count <= 3 {
            return limited
        } else if limited.count <= 6 {
            let areaCode = String(limited.prefix(3))
            let rest = String(limited.dropFirst(3))
            return "\(areaCode)-\(rest)"
        } else {
            let areaCode = String(limited.prefix(3))
            let middle = String(limited.dropFirst(3).prefix(3))
            let last = String(limited.dropFirst(6))
            return "\(areaCode)-\(middle)-\(last)"
        }
    }
}

// Custom TextField Style
struct CustomTextFieldStyle: TextFieldStyle {
    func _body(configuration: TextField<Self._Label>) -> some View {
        configuration
            .padding()
            .background(Color.white.opacity(0.95))
            .cornerRadius(10)
            .foregroundColor(.black)
            .autocapitalization(.none)
            .disableAutocorrection(true)
    }
}

#Preview {
    LoginView()
}
