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

    private var foregroundWindowScene: UIWindowScene? {
        let windowScenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        return windowScenes.first(where: { $0.activationState == .foregroundActive }) ?? windowScenes.first
    }

    /// Lock to portrait only
    func lockToPortrait() {
        lockedOrientation = .portrait

        // Force orientation update on iOS 16+
        if #available(iOS 16.0, *) {
            let windowScene = foregroundWindowScene
            windowScene?.keyWindow?.rootViewController?.setNeedsUpdateOfSupportedInterfaceOrientations()
            windowScene?.requestGeometryUpdate(.iOS(interfaceOrientations: .portrait)) { error in
                logDebug("⚠️ [Orientation] Portrait geometry update failed: \(error.localizedDescription)")
            }
        } else {
            UIDevice.current.setValue(UIInterfaceOrientation.portrait.rawValue, forKey: "orientation")
        }
    }

    /// Lock to landscape only
    func lockToLandscape() {
        lockedOrientation = .landscape

        // Request either landscape side so iOS can follow how the user is holding the phone.
        // Forcing landscapeRight caused a 180° flip for landscapeLeft captures.
        if #available(iOS 16.0, *) {
            let windowScene = foregroundWindowScene
            let requestedMask: UIInterfaceOrientationMask
            switch UIDevice.current.orientation {
            case .landscapeLeft:
                // Device and interface landscape names are intentionally opposite in UIKit.
                requestedMask = .landscapeRight
            case .landscapeRight:
                requestedMask = .landscapeLeft
            default:
                requestedMask = .landscape
            }
            windowScene?.keyWindow?.rootViewController?.setNeedsUpdateOfSupportedInterfaceOrientations()
            windowScene?.requestGeometryUpdate(.iOS(interfaceOrientations: requestedMask)) { error in
                logDebug("⚠️ [Orientation] Landscape geometry update failed: \(error.localizedDescription)")
            }
        } else {
            let interfaceOrientation: UIInterfaceOrientation = UIDevice.current.orientation == .landscapeLeft
                ? .landscapeRight
                : .landscapeLeft
            UIDevice.current.setValue(interfaceOrientation.rawValue, forKey: "orientation")
        }
    }

    /// Unlock to allow all orientations
    func unlock() {
        lockedOrientation = .all

        if #available(iOS 16.0, *) {
            let windowScene = foregroundWindowScene
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
        logDebug("🔥 [AppDelegate] didFinishLaunching START")

        FirebaseConfiguration.shared.setLoggerLevel(.error)
        logDebug("🔥 [AppDelegate] Firebase logging level set to .error")

        if FirebaseApp.app() == nil {
            logDebug("🔥 [AppDelegate] Configuring Firebase...")
            FirebaseApp.configure()
            logDebug("🔥 [AppDelegate] Firebase configured. App: \(FirebaseApp.app()?.name ?? "nil")")
        } else {
            logDebug("🔥 [AppDelegate] Firebase already configured")
        }

        logDebug("🔥 [AppDelegate] Auth settings: \(String(describing: Auth.auth().settings))")
        logDebug("🔥 [AppDelegate] didFinishLaunching END")

        application.registerForRemoteNotifications()
        logDebug("🔥 [AppDelegate] Requested APNs registration for Firebase Phone Auth")

        return true
    }

    // URL handling for phone auth (reCAPTCHA) is done in SceneDelegate.scene(_:openURLContexts:)
    // to avoid OpenURLOptionsKey deprecation in iOS 26.

    // Required by Firebase Auth - forward APNs token
    func application(_ application: UIApplication, didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data) {
        logDebug("🔥 [AppDelegate] didRegisterForRemoteNotifications - token received")

        // Forward to Firebase Auth - use safe method
        Auth.auth().setAPNSToken(deviceToken, type: .unknown)

        logDebug("🔥 [AppDelegate] APNs token forwarded to Firebase")
    }

    func application(_ application: UIApplication, didFailToRegisterForRemoteNotificationsWithError error: Error) {
        logDebug("🔥 [AppDelegate] didFailToRegisterForRemoteNotifications: \(error.localizedDescription)")
    }

    // Required by Firebase Auth - forward notifications
    func application(_ application: UIApplication, didReceiveRemoteNotification userInfo: [AnyHashable: Any], fetchCompletionHandler completionHandler: @escaping (UIBackgroundFetchResult) -> Void) {
        logDebug("🔥 [AppDelegate] didReceiveRemoteNotification")

        if Auth.auth().canHandleNotification(userInfo) {
            logDebug("🔥 [AppDelegate] Firebase handled the notification")
            completionHandler(.noData)
            return
        }

        logDebug("🔥 [AppDelegate] Firebase did NOT handle the notification")
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
            #if DEBUG
            if ProcessInfo.processInfo.arguments.contains("-PaafektScreenshotHome") {
                HomeViewWithBottomBar(authManager: authManager)
                    .onAppear {
                        authManager.enableScreenshotUser()
                    }
            } else if authManager.isAuthenticated {
                HomeViewWithBottomBar(authManager: authManager)
            } else {
                LoginView()
            }
            #else
            if authManager.isAuthenticated {
                HomeViewWithBottomBar(authManager: authManager)
            } else {
                LoginView()
            }
            #endif
        }
    }
}
