import SwiftUI
import FirebaseAuth

/// Debug view to test Firebase Phone Auth
/// Shows detailed status of each step in the authentication process
struct PhoneAuthDebugView: View {
    @StateObject private var authManager = AuthenticationManager()
    @State private var phoneNumber = "+1"
    @State private var otpCode = ""
    @State private var logs: [String] = []
    @State private var showingAlert = false
    @State private var alertMessage = ""
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Phone Number Input
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Phone Number")
                            .font(.headline)
                        TextField("Enter phone number with country code", text: $phoneNumber)
                            .textFieldStyle(.roundedBorder)
                            .keyboardType(.phonePad)
                        Text("Example: +1 650-555-3434")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    // Send OTP Button
                    Button(action: sendOTP) {
                        HStack {
                            if authManager.isLoading {
                                ProgressView()
                                    .scaleEffect(0.8)
                            }
                            Text(authManager.isLoading ? "Sending..." : "Send OTP")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                    .disabled(phoneNumber.count < 10 || authManager.isLoading)
                    
                    Divider()
                    
                    // OTP Verification
                    VStack(alignment: .leading, spacing: 8) {
                        Text("OTP Code")
                            .font(.headline)
                        TextField("Enter 6-digit code", text: $otpCode)
                            .textFieldStyle(.roundedBorder)
                            .keyboardType(.numberPad)
                    }
                    
                    Button(action: verifyOTP) {
                        Text("Verify OTP")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(otpCode.count == 6 ? Color.green : Color.gray)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    .disabled(otpCode.count != 6 || authManager.isLoading)
                    
                    Divider()
                    
                    // Status & Logs
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Debug Logs")
                            .font(.headline)
                        
                        if authManager.isAuthenticated {
                            HStack {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                                Text("Authenticated!")
                                    .fontWeight(.semibold)
                            }
                        }
                        
                        if let error = authManager.errorMessage {
                            HStack(alignment: .top) {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .foregroundColor(.orange)
                                Text(error)
                                    .font(.caption)
                            }
                            .padding(8)
                            .background(Color.orange.opacity(0.1))
                            .cornerRadius(8)
                        }
                        
                        ScrollView {
                            VStack(alignment: .leading, spacing: 4) {
                                ForEach(Array(logs.enumerated()), id: \.offset) { index, log in
                                    Text("[\(index + 1)] \(log)")
                                        .font(.system(.caption, design: .monospaced))
                                        .foregroundColor(.secondary)
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .frame(height: 200)
                        .padding(8)
                        .background(Color.secondary.opacity(0.1))
                        .cornerRadius(8)
                    }
                    
                    // Clear button
                    Button("Clear Logs") {
                        logs.removeAll()
                    }
                    .foregroundColor(.red)
                }
                .padding()
            }
            .navigationTitle("Phone Auth Test")
            .alert("Status", isPresented: $showingAlert) {
                Button("OK", role: .cancel) { }
            } message: {
                Text(alertMessage)
            }
        }
    }
    
    private func addLog(_ message: String) {
        let timestamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        logs.append("[\(timestamp)] \(message)")
    }
    
    private func sendOTP() {
        addLog("📤 Sending OTP to: \(phoneNumber)")
        
        authManager.sendOTP(to: phoneNumber) { success, error in
            if success {
                addLog("✅ OTP sent successfully!")
                addLog("📱 Check your phone for SMS code")
                alertMessage = "OTP sent! Check your phone for the verification code."
                showingAlert = true
            } else {
                let errorMsg = error ?? "Unknown error"
                addLog("❌ Error sending OTP: \(errorMsg)")
                
                // Special handling for error 17054
                if errorMsg.contains("17054") || errorMsg.contains("swizzling") {
                    addLog("ℹ️  Error 17054 is normal on iOS Simulator")
                    addLog("ℹ️  SMS should still be sent - wait 30-60 seconds")
                    addLog("💡 To fix: Add 'FirebaseAppDelegateProxyEnabled=true' to Info.plist")
                    alertMessage = """
                    APNs not configured (normal on Simulator).
                    
                    SMS should still arrive in 30-60 seconds.
                    Check your phone!
                    """
                } else {
                    alertMessage = "Error: \(errorMsg)"
                }
                showingAlert = true
            }
        }
    }
    
    private func verifyOTP() {
        addLog("🔐 Verifying OTP: \(otpCode)")
        
        authManager.verifyOTP(otpCode, name: "Test User", phoneNumber: phoneNumber) { success, error in
            if success {
                addLog("✅ OTP verified successfully!")
                addLog("🎉 Authentication complete!")
                alertMessage = "Success! You are now authenticated."
                showingAlert = true
            } else {
                let errorMsg = error ?? "Invalid code"
                addLog("❌ Verification failed: \(errorMsg)")
                alertMessage = "Verification failed: \(errorMsg)"
                showingAlert = true
            }
        }
    }
}

#Preview {
    PhoneAuthDebugView()
}
