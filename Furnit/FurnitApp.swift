import SwiftUI

// Option 1: Your current approach (passing as parameter)
@main
struct FurnitApp: App {
    @StateObject private var authManager = AuthenticationManager()
    @StateObject private var appStateManager = AppStateManager.shared
    
    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(authManager)
                .environmentObject(appStateManager)
                .preferredColorScheme(.dark)
        }
    }
}

struct RootView: View {
    @EnvironmentObject var authManager: AuthenticationManager
    
    var body: some View {
        Group {
            if authManager.isAuthenticated {
                // This now works - HomeViewWithBottomBar accepts authManager parameter
                HomeViewWithBottomBar(authManager: authManager)
            } else {
                LoginView()
            }
        }
    }
}

// Option 2: Alternative using @EnvironmentObject (cleaner approach)
// If you prefer not to pass parameters, you can change HomeViewWithBottomBar to use:
/*
struct HomeViewWithBottomBar: View {
    @EnvironmentObject var authManager: AuthenticationManager
    @StateObject private var roomManager = RoomManager.shared
    // ... rest of the view
}

// Then in RootView, just call it without parameters:
struct RootView: View {
    @EnvironmentObject var authManager: AuthenticationManager
    
    var body: some View {
        Group {
            if authManager.isAuthenticated {
                HomeViewWithBottomBar()  // No parameter needed
            } else {
                LoginView()
            }
        }
    }
}
*/
