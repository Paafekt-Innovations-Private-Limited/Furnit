import SwiftUI

// In FurnitApp.swift
@main
struct FurnitApp: App {
    @StateObject private var authManager = AuthenticationManager()
    @StateObject private var appStateManager = AppStateManager.shared
    
    var body: some Scene {
        WindowGroup {
            RootView() // Use a wrapper instead
                .environmentObject(authManager)
                .environmentObject(appStateManager)
                .preferredColorScheme(.dark)
        }
    }
}

// Create a new RootView
struct RootView: View {
    @EnvironmentObject var authManager: AuthenticationManager
    
    var body: some View {
        Group {
            if authManager.isAuthenticated {
                HomeViewWithProfile(authManager: authManager)
            } else {
                LoginView()
            }
        }
    }
}
