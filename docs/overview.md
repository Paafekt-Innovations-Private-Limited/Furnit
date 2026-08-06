# Furnit overview

Furnit is a cross-platform mobile product for turning room photos into viewable 3D
rooms and checking how detected furniture fits inside them. Core room processing,
camera calibration, and furniture inference are designed to run on device.

- Active context: [`READ_FIRST.md`](READ_FIRST.md)
- Full documentation index: [`README.md`](README.md)
- Architecture entry point: [`architecture.md`](architecture.md)

## Product surfaces

| Surface | What it does | Primary implementation |
|---|---|---|
| Phone sign-in | Firebase phone-number authentication, country picker, OTP verification | iOS `Furnit/Authentication/`; Android `android/app/src/main/java/com/furnit/android/auth/` |
| Photo → 3D | Creates a preview, measures the room, and saves a local 3D asset | iOS `Furnit/Services/RoomReconstruction/`; Android `android/app/src/main/java/com/furnit/android/services/` and `roomreconstruction/` |
| Room viewer | Opens saved USDZ/GLB/mesh/splat content with measurement and camera controls | iOS `Furnit/Views/`; Android `GLBRoomActivity` and viewer assets |
| Furniture Fit | Detects/segments furniture and composites movable cutouts over the room | iOS RTMDet FP16 LiteRT (mandatory audited full Metal); Android RTMDet FP16 LiteRT (GPU or XNNPACK CPU) |
| Local library | Stores and reopens saved rooms on the device | iOS `Documents/SavedRooms`; Android `files/rooms` |
| Licenses and attributions | Shows bundled notices and Apache text without requiring a network connection | iOS `Furnit/Licenses/` + `LicensesView`; Android `app/src/main/assets/legal/` + `LicensesActivity` |

## Platform outputs

| Platform | Default single-photo result | Viewer/runtime |
|---|---|---|
| iOS | Immediate flat preview; textured metric-context USDZ on first save | SwiftUI/UIKit + RealityKit/SceneKit/MetalSplatter |
| Android | Flat full-photo GLB preview; manual five-plane GLB alternative | Kotlin + WebView/Three.js, with SceneView/ARCore where needed |

The implementations aim for equivalent product behavior, but their model formats,
renderers, and persistence details are platform-native. Cross-platform parity should
be defined at the user-flow/metadata level, not by forcing identical internal code.

## Repository map

| Path | Ownership |
|---|---|
| [`../Furnit/`](../Furnit/) | iOS application and code-local docs |
| [`../android/`](../android/) | Android application and code-local docs |
| [`../scripts/`](../scripts/) | iOS/shared model conversion and verification tools |
| [`../android/scripts/`](../android/scripts/) | Android/Firebase maintenance scripts |
| [`../docs/`](.) | Shared architecture, policy, release, research, and runbook docs |

Paafekt Innovations Private Limited (India) operates and publishes the mobile App.
Paafekt Inc. (United States) is an affiliate only. See
[`privacy.html`](privacy.html) for the repository privacy-policy source.
