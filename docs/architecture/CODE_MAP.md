# Furnit code map

This map points from product behavior to the code that owns it. Paths are relative to
the repository root. Confirm implementation details in code before changing a dated
document.

## Shared repository surfaces

| Path | Responsibility |
|---|---|
| `README.md`, `AGENTS.md`, `docs/READ_FIRST.md` | Golden-path entry points |
| `docs/` | Shared architecture, operations, legal-policy source, release state, research, and history |
| `scripts/` | Model export, conversion, localization sync, and verification tools |
| `Furnit.xcodeproj`, `Furnit.xcworkspace` | iOS build graph and Swift package resolution |
| `android/settings.gradle`, `android/app/build.gradle` | Android modules, package identity, SDK targets, asset packs, dependencies, and signing |

## iOS application

| Area | Owning code |
|---|---|
| App shell/navigation | `Furnit/FurnitApp.swift`, `Furnit/Views/ContentView.swift` |
| Phone login | `Furnit/Authentication/LoginView.swift`, `AuthenticationManager.swift`, `OTPVerificationView.swift` |
| Photo → 3D UI and two-phase dispatch | `Furnit/Views/Components/SinglePhotoRoomViewer.swift` |
| Default metric room reconstruction | `Furnit/Services/RoomReconstruction/DepthAnythingRoomReconstructor.swift`, `GeoCalibCalibrationService.swift`, `FocalResolver.swift`, `ScaleEstimator.swift`, `RoomHeight.swift`, `RoomExtent.swift` |
| Manual boundary reconstruction | `Furnit/Services/RoomReconstruction/SinglePhotoRoomReconstructor.swift`, `SyntheticDepthEstimator.swift` |
| Capture/calibration sidecars | `Furnit/Services/OnDevice/CameraExifSidecar.swift`, `RoomGenerationCameraSidecar.swift`, `Furnit/Views/Components/ARRoomPhotoCaptureView.swift` |
| RTMDet model/inference | `Furnit/Services/OnDevice/RTMDetModelService.swift`, `RTMDetImageInference.swift` |
| Furniture Fit UI/selection/overlays | `Furnit/Views/FurnitureFit/` |
| USDZ/GLB/mesh/splat viewers | `Furnit/Views/ModelViewerView.swift`, `GLBRoomView.swift`, `MeshRoomView.swift`, `SplatRoomView.swift`, `Furnit/Views/Components/RealityKitView.swift`, `GaussianSplatView.swift` |
| Theme/brand | `Furnit/Theme/Theme.swift`, `Furnit/Assets.xcassets/`, `docs/PAAFEKT_DESIGN_SYSTEM.md` |
| Privacy manifest/config | `Furnit/PrivacyInfo.xcprivacy`, `Furnit/Info.plist`, `Furnit/Furnit.entitlements` |
| Tests | `FurnitTests/` |

iOS detail starts at `Furnit/docs/README.md`; RTMDet detail is owned by
`Furnit/Views/FurnitureFit/README.md`.

## Android application

| Area | Owning code |
|---|---|
| App shell/navigation | `android/app/src/main/java/com/furnit/android/FurnitApplication.kt`, `ContentActivity.kt` |
| Login/country/OTP | `android/app/src/main/java/com/furnit/android/auth/LoginActivity.kt`, `CountryCode.kt`, `AuthenticationManager.kt`, `OTPVerificationActivity.kt` |
| Manifest/app identity | `android/app/src/main/AndroidManifest.xml`, `android/app/build.gradle`, `android/app/google-services.json` |
| Photo → 3D UI | `android/app/src/main/java/com/furnit/android/SinglePhotoRoomActivity.kt` |
| Generation lifecycle | `android/app/src/main/java/com/furnit/android/services/PhotoRoomGenerationService.kt`, `SinglePhotoRoomReconstructor.kt`, `GlbGenerator.kt` |
| Metric measurement/calibration | `android/app/src/main/java/com/furnit/android/services/DepthAnythingRoomMeasurer.kt`, `roomreconstruction/DepthAnythingRoomMeasurementPipeline.kt`, `GeoCalibCalibrationService.kt`, `FocalResolver.kt`, `ScaleEstimator.kt` |
| Manual boundaries | `android/app/src/main/java/com/furnit/android/RoomBoundaryActivity.kt`, `utils/RoomBoundaryManager.kt` |
| GLB viewer/library | `android/app/src/main/java/com/furnit/android/GLBRoomActivity.kt`, `models/ModelManager.kt`, WebView assets under `android/app/src/main/assets/` |
| RTMDet/Furniture Fit | `android/app/src/main/java/com/furnit/android/services/FurnitureFitManager.kt`, `FurnitureFitOverlayView.kt`, `FurnitureFitFragment.kt`, `FurnitureFitActivity.kt` |
| Packaged models | `android/room_generation_models/src/main/assets/`, `android/rtmdet_models/src/main/assets/` |
| Firebase certificate repair | `android/scripts/fix_firebase_android_auth.sh` |
| Tests | `android/app/src/test/`, `android/app/src/androidTest/` |

Android detail starts at `android/docs/README.md`. Production login/signing detail is
owned by `android/docs/AUTHENTICATION.md`.

## Data and artifact boundaries

| Artifact | Location/contract |
|---|---|
| iOS saved rooms | App-private `Documents/SavedRooms` |
| Android previews | App-private `files/room_previews/<room-id>/` |
| Android saved rooms | App-private `files/rooms/<room-id>/` |
| Android generated room | `room.glb` plus `metadata.txt` |
| Android model delivery | Install-time Play Asset Delivery modules `room_generation_models` and `rtmdet_models` |
| Firebase config | `Furnit/Authentication/GoogleService-Info.plist`, `android/app/google-services.json` |

Never commit local signing credentials, keystores, downloaded tokens, or generated
private user content.
