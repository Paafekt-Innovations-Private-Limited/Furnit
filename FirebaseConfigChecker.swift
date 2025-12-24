import Foundation
import FirebaseCore

/// Diagnostic tool to check Firebase configuration
/// Run this in your app to verify all required values are present
class FirebaseConfigChecker {
    
    static func diagnose() {
        print("\n=== 🔍 Firebase Configuration Diagnostics ===\n")
        
        // Check if GoogleService-Info.plist exists
        if let path = Bundle.main.path(forResource: "GoogleService-Info", ofType: "plist") {
            print("✅ GoogleService-Info.plist found at: \(path)")
            
            // Read the plist
            if let plistData = FileManager.default.contents(atPath: path),
               let plist = try? PropertyListSerialization.propertyList(from: plistData, format: nil) as? [String: Any] {
                
                print("\n📋 Plist Contents:")
                
                // Check required keys
                let requiredKeys = [
                    "API_KEY",
                    "GCM_SENDER_ID",
                    "PLIST_VERSION",
                    "BUNDLE_ID",
                    "PROJECT_ID",
                    "STORAGE_BUCKET",
                    "IS_ADS_ENABLED",
                    "IS_ANALYTICS_ENABLED",
                    "IS_APPINVITE_ENABLED",
                    "IS_GCM_ENABLED",
                    "IS_SIGNIN_ENABLED",
                    "GOOGLE_APP_ID",
                    "REVERSED_CLIENT_ID",  // ⚠️ THIS IS THE KEY ONE FOR PHONE AUTH
                    "CLIENT_ID"
                ]
                
                for key in requiredKeys {
                    if let value = plist[key] {
                        if key == "API_KEY" || key == "REVERSED_CLIENT_ID" || key == "CLIENT_ID" {
                            // Show partial value for security
                            let stringValue = "\(value)"
                            let partial = String(stringValue.prefix(20)) + "..."
                            print("  ✅ \(key): \(partial)")
                        } else {
                            print("  ✅ \(key): \(value)")
                        }
                    } else {
                        print("  ❌ \(key): MISSING")
                    }
                }
                
                // Check REVERSED_CLIENT_ID specifically
                if let reversedClientID = plist["REVERSED_CLIENT_ID"] as? String {
                    print("\n🔑 REVERSED_CLIENT_ID: \(reversedClientID)")
                    print("   👉 Add this to Info.plist → URL Types → URL Schemes")
                } else {
                    print("\n❌ REVERSED_CLIENT_ID is MISSING!")
                    print("   This is required for Phone Authentication")
                    print("   Download a fresh GoogleService-Info.plist from Firebase Console")
                }
            }
        } else {
            print("❌ GoogleService-Info.plist NOT FOUND in bundle!")
        }
        
        // Check Firebase App configuration
        print("\n🔧 Firebase App Status:")
        if let app = FirebaseApp.app() {
            print("  ✅ FirebaseApp initialized: \(app.name)")
            print("  API Key: \(app.options.apiKey?.isEmpty == false ? "✅ Set" : "❌ Missing")")
            print("  Google App ID: \(app.options.googleAppID.isEmpty ? "❌ Missing" : "✅ \(app.options.googleAppID)")")
            print("  Project ID: \(app.options.projectID?.isEmpty == false ? "✅ Set" : "❌ Missing")")
            print("  Client ID: \(app.options.clientID?.isEmpty == false ? "✅ Set" : "❌ Missing")")
            print("  GCM Sender ID: \(app.options.gcmSenderID.isEmpty ? "❌ Missing" : "✅ Set")")
            
            if app.options.clientID?.isEmpty != false {
                print("\n⚠️  CLIENT_ID is missing!")
                print("   This causes the PhoneAuthProvider crash")
                print("   Solution: Re-download GoogleService-Info.plist from Firebase Console")
            }
        } else {
            print("  ❌ FirebaseApp NOT configured")
        }
        
        // Check URL Schemes
        print("\n🔗 Registered URL Schemes:")
        if let urlTypes = Bundle.main.infoDictionary?["CFBundleURLTypes"] as? [[String: Any]] {
            for (index, urlType) in urlTypes.enumerated() {
                print("  URL Type \(index + 1):")
                if let schemes = urlType["CFBundleURLSchemes"] as? [String] {
                    for scheme in schemes {
                        print("    - \(scheme)")
                    }
                }
            }
            
            // Check if REVERSED_CLIENT_ID is registered
            let allSchemes = urlTypes.flatMap { urlType -> [String] in
                return (urlType["CFBundleURLSchemes"] as? [String]) ?? []
            }
            
            if let path = Bundle.main.path(forResource: "GoogleService-Info", ofType: "plist"),
               let plistData = FileManager.default.contents(atPath: path),
               let plist = try? PropertyListSerialization.propertyList(from: plistData, format: nil) as? [String: Any],
               let reversedClientID = plist["REVERSED_CLIENT_ID"] as? String {
                
                if allSchemes.contains(reversedClientID) {
                    print("\n  ✅ REVERSED_CLIENT_ID is registered as URL scheme")
                } else {
                    print("\n  ❌ REVERSED_CLIENT_ID is NOT registered as URL scheme!")
                    print("     This will cause PhoneAuthProvider to crash")
                    print("     Add '\(reversedClientID)' to Info.plist → URL Types")
                }
            }
        } else {
            print("  ❌ No URL types registered")
            print("     You need to add the REVERSED_CLIENT_ID as a URL scheme")
        }
        
        print("\n=== End Diagnostics ===\n")
    }
}
