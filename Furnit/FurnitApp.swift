import SwiftUI
import FirebaseCore
import FirebaseAuth
import FirebaseAppCheck
import UserNotifications

// App Check provider factory for bot protection
class AppCheckProviderFactory: NSObject, FirebaseAppCheck.AppCheckProviderFactory {
    func createProvider(with app: FirebaseApp) -> AppCheckProvider? {
        #if DEBUG
        // Use debug provider for simulator/development
        // Check Xcode console for: "App Check debug token: XXXX"
        // Add this token to Firebase Console → App Check → Apps → Manage debug tokens
        print("🔐 [AppCheck] Using DEBUG provider - check console for debug token")
        return AppCheckDebugProvider(app: app)
        #else
        // Use App Attest for production (real devices)
        return AppAttestProvider(app: app)
        #endif
    }
}

class AppDelegate: NSObject, UIApplicationDelegate, UNUserNotificationCenterDelegate {
    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        // Enable App Check for bot protection (must be before FirebaseApp.configure)
        let providerFactory = AppCheckProviderFactory()
        AppCheck.setAppCheckProviderFactory(providerFactory)

        FirebaseApp.configure()

        // Request notification permissions for phone auth
        UNUserNotificationCenter.current().delegate = self
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) { granted, error in
            if granted {
                DispatchQueue.main.async {
                    application.registerForRemoteNotifications()
                }
            }
        }

        return true
    }

    // Handle URL schemes for phone auth (reCAPTCHA)
    func application(_ app: UIApplication, open url: URL, options: [UIApplication.OpenURLOptionsKey: Any] = [:]) -> Bool {
        if Auth.auth().canHandle(url) {
            return true
        }
        return false
    }

    // Forward APNs token to Firebase
    func application(_ application: UIApplication, didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data) {
        Auth.auth().setAPNSToken(deviceToken, type: .unknown)
    }

    // Forward remote notifications to Firebase Auth
    func application(_ application: UIApplication, didReceiveRemoteNotification userInfo: [AnyHashable: Any], fetchCompletionHandler completionHandler: @escaping (UIBackgroundFetchResult) -> Void) {
        if Auth.auth().canHandleNotification(userInfo) {
            completionHandler(.noData)
            return
        }
        completionHandler(.noData)
    }
}

// Option 1: Your current approach (passing as parameter)
@main
struct FurnitApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var delegate
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
