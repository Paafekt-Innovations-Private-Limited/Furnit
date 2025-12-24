import SwiftUI
import FirebaseCore
import FirebaseAuth
import UserNotifications

class AppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        print("🔥 [AppDelegate] didFinishLaunching START")

        // Configure Firebase here - proper place for swizzling to work
        if FirebaseApp.app() == nil {
            print("🔥 [AppDelegate] Configuring Firebase...")
            FirebaseApp.configure()
            print("🔥 [AppDelegate] Firebase configured. App: \(FirebaseApp.app()?.name ?? "nil")")
        } else {
            print("🔥 [AppDelegate] Firebase already configured")
        }

        // Log Auth settings
        print("🔥 [AppDelegate] Auth settings: \(String(describing: Auth.auth().settings))")
        print("🔥 [AppDelegate] didFinishLaunching END")
        return true
    }

    // Handle URL schemes for phone auth (reCAPTCHA callback)
    func application(_ app: UIApplication, open url: URL, options: [UIApplication.OpenURLOptionsKey: Any] = [:]) -> Bool {
        print("🔥 [AppDelegate] open URL called: \(url.absoluteString)")
        let canHandle = Auth.auth().canHandle(url)
        print("🔥 [AppDelegate] Auth.canHandle(url) = \(canHandle)")
        if canHandle {
            return true
        }
        return false
    }

    // Required by Firebase Auth - forward APNs token
    func application(_ application: UIApplication, didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data) {
        print("🔥 [AppDelegate] didRegisterForRemoteNotifications - token received")
        // Forward to Firebase Auth - use safe method
        Auth.auth().setAPNSToken(deviceToken, type: .unknown)
        print("🔥 [AppDelegate] APNs token forwarded to Firebase")
    }

    func application(_ application: UIApplication, didFailToRegisterForRemoteNotificationsWithError error: Error) {
        print("🔥 [AppDelegate] didFailToRegisterForRemoteNotifications: \(error.localizedDescription)")
    }

    // Required by Firebase Auth - forward notifications
    func application(_ application: UIApplication, didReceiveRemoteNotification userInfo: [AnyHashable: Any], fetchCompletionHandler completionHandler: @escaping (UIBackgroundFetchResult) -> Void) {
        print("🔥 [AppDelegate] didReceiveRemoteNotification")
        if Auth.auth().canHandleNotification(userInfo) {
            print("🔥 [AppDelegate] Firebase handled the notification")
            completionHandler(.noData)
            return
        }
        print("🔥 [AppDelegate] Firebase did NOT handle the notification")
        completionHandler(.noData)
    }
}

@main
struct FurnitApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject private var authManager: AuthenticationManager
    @StateObject private var appStateManager = AppStateManager.shared

    init() {
        // AuthenticationManager will wait for Firebase to be configured
        _authManager = StateObject(wrappedValue: AuthenticationManager())
    }

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
                HomeViewWithBottomBar(authManager: authManager)
            } else {
                LoginView()
            }
        }
    }
}
