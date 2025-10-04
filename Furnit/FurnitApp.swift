import SwiftUI

@main
struct FurnitApp: App {
    @StateObject private var authManager = AuthenticationManager()
    @StateObject private var appStateManager = AppStateManager.shared
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(authManager)
                .environmentObject(appStateManager)
                .environment(\.appState, appStateManager)
        }
    }
}
