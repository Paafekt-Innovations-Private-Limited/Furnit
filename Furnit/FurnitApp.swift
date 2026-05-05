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
        logDebug("🔥 [AppDelegate] didFinishLaunching START")

        SHARPService.purgeTemporarySharpModelsDirectoryAtLaunch()

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

    /// Best-effort unload of heavy Core ML before Jetsam.
    /// Important: do **not** call ``SHARPService/releaseResources()`` here — that ends ODR access
    /// (`endAccessingResources`) for the ~1.2 GB tagged SHARP pack. Stopping Xcode often triggers a
    /// memory warning; releasing ODR makes the next standalone room creation fail until remount
    /// succeeds. Unloading only the in-memory ``MLModel`` keeps the pack mounted and avoids that trap.
    func applicationDidReceiveMemoryWarning(_ application: UIApplication) {
        Task { @MainActor in
            logDebug("⚠️ [AppDelegate] Memory warning — unloading SHARP/YOLOE Core ML (keeping ODR mounts)")
            YoloEDetectionParser.trimScratchBuffers()
            SHARPService.shared.releaseInferenceMemoryAfterGeneration()
            YOLOEModelService.shared.releaseLoadedModelOnlyPreservingODR()
        }
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
            if authManager.isAuthenticated {
                HomeViewWithBottomBar(authManager: authManager)
            } else {
                LoginView()
            }
        }
        .overlay(alignment: .bottom) {
            SharpGenerationBottomBar()
                .padding(.horizontal, 12)
                .padding(.bottom, 8)
                .zIndex(1000)
        }
    }
}
