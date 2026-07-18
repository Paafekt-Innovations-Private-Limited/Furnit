# Copilot / AI Agent Instructions for Furnit

Furnit contains native iOS and Android apps for single-photo room creation, room viewing, and on-device furniture detection/segmentation.

## Current architecture

- iOS entry point: `Furnit/FurnitApp.swift`; project/scheme: `Furnit.xcodeproj` / `Furnit`.
- Android entry point: `android/app`; open the `android/` directory in Android Studio.
- iOS room measurement uses Depth Anything, optional GeoCalib metadata, and RTMDet object anchors; saved rooms use USDZ.
- Android room measurement uses ONNX Runtime for Depth Anything and RTMDet, with optional GeoCalib; saved rooms use GLB.
- Furniture Fit uses RTMDet-Ins on both platforms.
- iOS splat viewing uses MetalSplatter. GLB/mesh viewers use bundled Three.js resources where available.

## Important locations

- `Furnit/Views/`, `Furnit/Services/`, `Furnit/Models/`, `Furnit/Utilities/`
- `Furnit/Models/DepthAnything/`, `Furnit/Models/GeoCalib/`, `Furnit/Models/RTMDet/`
- `android/app/src/main/java/com/furnit/android/`
- `android/app/src/main/assets/`
- `docs/` and `android/docs/`

Model and asset paths may be loaded dynamically. Search string paths and asset manifests before moving or renaming resources.

## Verification

iOS compile check:

```bash
xcodebuild -project "Furnit.xcodeproj" -scheme "Furnit" -configuration Debug \
  -sdk iphoneos -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO COMPILER_INDEX_STORE_ENABLE=NO ENABLE_PREVIEWS=NO \
  -jobs 2 build
```

Android compile check:

```bash
cd android
./gradlew :app:assembleDebug --no-daemon
```

Do not use an iOS Simulator for default verification. AR, camera, Core ML, Metal, and ONNX behavior still requires appropriate physical-device testing.

## Conventions

- Use descriptive identifiers.
- Keep lifecycle and resource ownership explicit.
- Preserve localization parity when adding or removing user-facing keys.
- Update nearby documentation when runtime paths, models, or assets change.
- Compile both affected platforms before concluding implementation work.
