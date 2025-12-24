import SwiftUI

/// Helper view that explains Firebase Test Phone Numbers vs Real Phone Numbers
/// Show this in your login view or as a help button
struct PhoneNumberTestingExplainer: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Title
            HStack {
                Image(systemName: "phone.fill")
                    .foregroundColor(.blue)
                Text("Testing Phone Authentication")
                    .font(.headline)
            }
            
            Divider()
            
            // Test Numbers Section
            VStack(alignment: .leading, spacing: 8) {
                Label("Firebase Test Numbers", systemImage: "hammer.fill")
                    .font(.subheadline)
                    .foregroundColor(.orange)
                
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 4) {
                        Text("•")
                        Text("No SMS is sent")
                    }
                    HStack(spacing: 4) {
                        Text("•")
                        Text("Use pre-configured code from Firebase Console")
                    }
                    HStack(spacing: 4) {
                        Text("•")
                        Text("Instant verification (no waiting)")
                    }
                    HStack(spacing: 4) {
                        Text("•")
                        Text("Example: +1 650-555-3434 → Code: 123456")
                            .italic()
                    }
                }
                .font(.caption)
                .foregroundColor(.secondary)
            }
            .padding(12)
            .background(Color.orange.opacity(0.1))
            .cornerRadius(8)
            
            // Real Numbers Section
            VStack(alignment: .leading, spacing: 8) {
                Label("Real Phone Numbers", systemImage: "message.fill")
                    .font(.subheadline)
                    .foregroundColor(.green)
                
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 4) {
                        Text("•")
                        Text("Real SMS is sent")
                    }
                    HStack(spacing: 4) {
                        Text("•")
                        Text("Wait 30-60 seconds for message")
                    }
                    HStack(spacing: 4) {
                        Text("•")
                        Text("Enter code from SMS")
                    }
                    HStack(spacing: 4) {
                        Text("•")
                        Text("Example: +1 415-123-4567 → Check SMS")
                            .italic()
                    }
                }
                .font(.caption)
                .foregroundColor(.secondary)
            }
            .padding(12)
            .background(Color.green.opacity(0.1))
            .cornerRadius(8)
            
            // Setup Instructions
            VStack(alignment: .leading, spacing: 8) {
                Label("How to Setup Test Numbers", systemImage: "gear")
                    .font(.subheadline)
                    .foregroundColor(.blue)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("1. Go to Firebase Console → Authentication")
                    Text("2. Sign-in method tab → Phone numbers for testing")
                    Text("3. Add test number + 6-digit verification code")
                    Text("4. Use that code in your app (no SMS needed)")
                }
                .font(.caption)
                .foregroundColor(.secondary)
            }
            .padding(12)
            .background(Color.blue.opacity(0.1))
            .cornerRadius(8)
            
            // Common Issue
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.yellow)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Common Confusion")
                        .font(.caption)
                        .fontWeight(.semibold)
                    Text("If you're using a test number and waiting for SMS, it will never arrive! Test numbers don't send real messages.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .padding(12)
            .background(Color.yellow.opacity(0.1))
            .cornerRadius(8)
        }
        .padding()
    }
}

/// Inline helper component for your login view
struct PhoneNumberTypeIndicator: View {
    let phoneNumber: String
    
    private var isTestNumber: Bool {
        // Common test number patterns
        let testPatterns = [
            "+1 650-555",
            "+16505555",
            "+1650555",
        ]
        
        return testPatterns.contains { phoneNumber.hasPrefix($0) }
    }
    
    var body: some View {
        if isTestNumber {
            HStack(spacing: 6) {
                Image(systemName: "hammer.circle.fill")
                    .foregroundColor(.orange)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Test Number Detected")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(.orange)
                    Text("No SMS will be sent. Use your Firebase Console code.")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.orange.opacity(0.15))
            .cornerRadius(8)
        } else {
            HStack(spacing: 6) {
                Image(systemName: "message.circle.fill")
                    .foregroundColor(.green)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Real Number")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(.green)
                    Text("SMS will be sent. Wait 30-60 seconds.")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.green.opacity(0.15))
            .cornerRadius(8)
        }
    }
}

#Preview("Explainer") {
    PhoneNumberTestingExplainer()
}

#Preview("Test Number Indicator") {
    VStack {
        PhoneNumberTypeIndicator(phoneNumber: "+1 650-555-3434")
        PhoneNumberTypeIndicator(phoneNumber: "+1 415-123-4567")
    }
    .padding()
}
