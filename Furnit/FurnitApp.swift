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

    /// Lock to landscape only
    func lockToLandscape() {
        lockedOrientation = .landscape

        // Force orientation update on iOS 16+
        if #available(iOS 16.0, *) {
            let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene
            windowScene?.requestGeometryUpdate(.iOS(interfaceOrientations: .landscapeRight))
            windowScene?.keyWindow?.rootViewController?.setNeedsUpdateOfSupportedInterfaceOrientations()
        } else {
            UIDevice.current.setValue(UIInterfaceOrientation.landscapeRight.rawValue, forKey: "orientation")
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

// MARK: - Scene delegate for open URL (iOS 26+ prefers UIScene lifecycle over OpenURLOptionsKey)
class SceneDelegate: NSObject, UIWindowSceneDelegate {
    func scene(_ scene: UIScene, openURLContexts URLContexts: Set<UIOpenURLContext>) {
        for context in URLContexts {
            if Auth.auth().canHandle(context.url) {
                return
            }
        }
    }
}

class AppDelegate: NSObject, UIApplicationDelegate {

    func application(_ application: UIApplication, configurationForConnecting connectingSceneSession: UISceneSession, options: UIScene.ConnectionOptions) -> UISceneConfiguration {
        let config = UISceneConfiguration(name: nil, sessionRole: connectingSceneSession.role)
        config.delegateClass = SceneDelegate.self
        return config
    }

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

        SHARPService.purgeTemporarySharpModelsDirectoryAtLaunch()

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

        application.registerForRemoteNotifications()
        if debugMode {
            print("🔥 [AppDelegate] Requested APNs registration for Firebase Phone Auth")
        }

        return true
    }

    /// Best-effort unload of heavy Core ML + ODR before Jetsam; restart also clears RAM, but this helps mid-session peaks.
    func applicationDidReceiveMemoryWarning(_ application: UIApplication) {
        Task { @MainActor in
            logDebug("⚠️ [AppDelegate] Memory warning — releasing SHARP + YOLOE")
            SHARPService.shared.releaseResources()
            YOLOEModelService.shared.releaseResources()
        }
    }

    // URL handling for phone auth (reCAPTCHA) is done in SceneDelegate.scene(_:openURLContexts:)
    // to avoid OpenURLOptionsKey deprecation in iOS 26.

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
        // Firebase is configured in AppDelegate so Phone Auth swizzling and APNs forwarding
        // are attached to the standard UIApplication lifecycle before OTP starts.
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
