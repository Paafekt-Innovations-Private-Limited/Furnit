//
//  FirebaseTestView.swift
//  Quick test view to verify Firebase setup
//

import SwiftUI
import FirebaseAuth
import FirebaseCore

struct FirebaseTestView: View {
    @State private var testResult = "Firebase not tested yet"
    @State private var isLoading = false
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Firebase Setup Test")
                .font(.title)
                .fontWeight(.bold)
            
            Text(testResult)
                .foregroundColor(testResult.contains("✅") ? .green : 
                               testResult.contains("❌") ? .red : .primary)
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(8)
            
            if isLoading {
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle())
            }
            
            Button("Test Firebase Connection") {
                testFirebaseSetup()
            }
            .buttonStyle(.borderedProminent)
            .disabled(isLoading)
            
            Text("Instructions:")
                .font(.headline)
                .padding(.top)
            
            VStack(alignment: .leading, spacing: 8) {
                Text("1. ✅ Add Firebase SDK via SPM")
                Text("2. ⏳ Add real GoogleService-Info.plist")
                Text("3. ⏳ Enable Phone Auth in Firebase Console")
                Text("4. ⏳ Test with real phone number")
            }
            .font(.caption)
            .foregroundColor(.secondary)
            
            Spacer()
        }
        .padding()
        .navigationTitle("Setup Test")
    }
    
    private func testFirebaseSetup() {
        isLoading = true
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            // Test if Firebase is properly initialized
            if let app = FirebaseApp.app() {
                testResult = "✅ Firebase initialized successfully!\nApp Name: \(app.name)"
            } else {
                testResult = "❌ Firebase not initialized. Check GoogleService-Info.plist"
            }
            isLoading = false
        }
    }
}

#Preview {
    NavigationView {
        FirebaseTestView()
    }
}
