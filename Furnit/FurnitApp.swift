import SwiftUI
import FirebaseCore
import FirebaseAuth
import UserNotifications

// MARK: - Orientation Lock Manager

/// Manages app-wide orientation locking for specific views
class OrientationLockManager {
    static let shared = OrientationLockManager()

    /// Currently allowed orientations (default: all)
    var lockedOrientation: UIInterfaceOrientationMask = .all

    private init() {}

    /// Lock to portrait only
    func lockToPortrait() {
        lockedOrientation = .portrait

        // Force orientation update on iOS 16+
        if #available(iOS 16.0, *) {
            let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene
            windowScene?.requestGeometryUpdate(.iOS(interfaceOrientations: .portrait))
            windowScene?.keyWindow?.rootViewController?.setNeedsUpdateOfSupportedInterfaceOrientations()
        } else {
            UIDevice.current.setValue(UIInterfaceOrientation.portrait.rawValue, forKey: "orientation")
        }
    }

    /// Unlock to allow all orientations
    func unlock() {
        lockedOrientation = .all

        if #available(iOS 16.0, *) {
            let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene
            windowScene?.keyWindow?.rootViewController?.setNeedsUpdateOfSupportedInterfaceOrientations()
        }
    }
}

class AppDelegate: NSObject, UIApplicationDelegate {

    // MARK: - Orientation Support

    func application(_ application: UIApplication, supportedInterfaceOrientationsFor window: UIWindow?) -> UIInterfaceOrientationMask {
        return OrientationLockManager.shared.lockedOrientation
    }

    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode
        
        if debugMode {
            print("🔥 [AppDelegate] didFinishLaunching START")
        }

        // ✅ Configure Firebase logging level based on debug mode
        if debugMode {
            // Debug mode ON: Show all Firebase logs
            FirebaseConfiguration.shared.setLoggerLevel(.error)
            print("🔥 [AppDelegate] Firebase logging: DEBUG (all logs enabled)")
        } else {
            // Debug mode OFF: Suppress all Firebase logs (including AppCheck)
            FirebaseConfiguration.shared.setLoggerLevel(.error)
        }

        // Configure Firebase here - proper place for swizzling to work
        if FirebaseApp.app() == nil {
            if debugMode {
                print("🔥 [AppDelegate] Configuring Firebase...")
            }
            FirebaseApp.configure()
            if debugMode {
                print("🔥 [AppDelegate] Firebase configured. App: \(FirebaseApp.app()?.name ?? "nil")")
            }
        } else {
            if debugMode {
                print("🔥 [AppDelegate] Firebase already configured")
            }
        }

        // Log Auth settings only in debug mode
        if debugMode {
            print("🔥 [AppDelegate] Auth settings: \(String(describing: Auth.auth().settings))")
            print("🔥 [AppDelegate] didFinishLaunching END")
        }

        return true
    }

    // Handle URL schemes for phone auth (reCAPTCHA callback)
    func application(_ app: UIApplication, open url: URL, options: [UIApplication.OpenURLOptionsKey: Any] = [:]) -> Bool {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode
        
        if debugMode {
            print("🔥 [AppDelegate] open URL called: \(url.absoluteString)")
        }
        
        let canHandle = Auth.auth().canHandle(url)
        
        if debugMode {
            print("🔥 [AppDelegate] Auth.canHandle(url) = \(canHandle)")
        }
        
        if canHandle {
            return true
        }
        return false
    }

    // Required by Firebase Auth - forward APNs token
    func application(_ application: UIApplication, didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data) {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode
        
        if debugMode {
            print("🔥 [AppDelegate] didRegisterForRemoteNotifications - token received")
        }
        
        // Forward to Firebase Auth - use safe method
        Auth.auth().setAPNSToken(deviceToken, type: .unknown)
        
        if debugMode {
            print("🔥 [AppDelegate] APNs token forwarded to Firebase")
        }
    }

    func application(_ application: UIApplication, didFailToRegisterForRemoteNotificationsWithError error: Error) {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode
        
        if debugMode {
            print("🔥 [AppDelegate] didFailToRegisterForRemoteNotifications: \(error.localizedDescription)")
        }
    }

    // Required by Firebase Auth - forward notifications
    func application(_ application: UIApplication, didReceiveRemoteNotification userInfo: [AnyHashable: Any], fetchCompletionHandler completionHandler: @escaping (UIBackgroundFetchResult) -> Void) {
        let debugMode = AppStateManager.shared.qualitySettings.debugMode
        
        if debugMode {
            print("🔥 [AppDelegate] didReceiveRemoteNotification")
        }
        
        if Auth.auth().canHandleNotification(userInfo) {
            if debugMode {
                print("🔥 [AppDelegate] Firebase handled the notification")
            }
            completionHandler(.noData)
            return
        }
        
        if debugMode {
            print("🔥 [AppDelegate] Firebase did NOT handle the notification")
        }
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
                .crashReportAlert()
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
