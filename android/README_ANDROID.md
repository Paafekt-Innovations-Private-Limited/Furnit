# Furnit Android

This folder contains the Android app for Furnit: photo/manual room creation, GLB room viewing, and on-device furniture segmentation/cutout.

## Quick Start

1. Open the `android` folder in Android Studio.
2. Let Gradle sync and install any missing Android SDK components.
3. Select an arm64 Android device.
4. Run the `app` configuration.

Terminal build:

```bash
./gradlew :app:assembleDebug
```

## Release Build and Signing

Play only accepts a signed App Bundle. The release signing config reads these four
values from Gradle properties or environment variables (nothing is committed):

- `PAAFEKT_UPLOAD_STORE_FILE`
- `PAAFEKT_UPLOAD_STORE_PASSWORD`
- `PAAFEKT_UPLOAD_KEY_ALIAS`
- `PAAFEKT_UPLOAD_KEY_PASSWORD`

One-time upload keystore creation (keep the file and passwords in a password
manager; losing the upload key means a reset request through Play support):

```bash
keytool -genkeypair -v \
  -keystore ~/keystores/paafekt-upload.jks \
  -alias paafekt-upload \
  -keyalg RSA -keysize 2048 -validity 10000
```

Put the four values in `~/.gradle/gradle.properties` (machine-global, outside the
repo), then build:

```bash
./gradlew :app:bundleRelease
```

Output: `app/build/outputs/bundle/release/app-release.aab`.

Per-release checklist:

- Archive `app/build/outputs/mapping/release/mapping.txt` and upload it with the
  release in Play Console. Release builds are R8-minified, so crash reports
  (including ones users email from `CrashReportActivity`) must be retraced with
  this file (`$ANDROID_HOME/cmdline-tools/latest/bin/retrace mapping.txt trace.txt`).
- After enrolling in Play App Signing, add the **Play App Signing key's SHA-256**
  (Play Console > Test and release > App integrity) to the Firebase project's
  Android app settings. Without it, Firebase phone auth fails in the store build
  even though local builds work.
- Bump `versionCode` in `app/build.gradle` for every upload.

## Room Creation

- `SinglePhotoRoomActivity` owns the take-photo/select-photo screen.
- The method picker is displayed immediately after image selection; sampled EXIF-aware bitmap decode and AI generation continue off the UI thread.
- `PhotoRoomGenerationService` owns the AI room-generation job lifecycle.
- `SinglePhotoRoomReconstructor` writes the textured GLB room output.
- `GlbGenerator.generateFlatPhotoGlb` builds the **AI default**: one full-photo textured plane (Swift parity, avoids dragged/stretched cuboid crops).
- AI flat-photo GLBs embed the full-photo texture as JPEG to reduce file size and speed up preview/save.
- `GlbGenerator.generateGlb` builds the **manual default**: floor, ceiling, and wall planes with cropped photo textures.
- `RoomGenerationAssets` records the packaged Swift-parity asset contract.
- Generated preview folders are promoted into `files/rooms/` only when the user saves the room.

Android currently packages the Depth Anything metric depth ONNX asset and the existing RTMDet furniture segmentation ONNX asset. GeoCalib needs an Android export before the full Swift metric reconstruction path can be wired.

## Furniture Segmentation

- Model: `rtmdet-ins-m-raw.onnx` in app assets.
- Runtime: ONNX Runtime Android.
- Manager: `app/src/main/java/com/furnit/android/services/FurnitureFitManager.kt`.
- Backend lifetime: shared process-wide `OrtEnvironment`, `OrtSession`, session options, and single-thread executor.
- Fast identify: live identify requests cls/bbox outputs only; mask kernels, `mask_feat`, mask planes, and affinity grouping are reserved for segmentation modes.
- Output: `SegmentationResult.mask` is an ARGB bitmap. Non-furniture pixels have alpha `0`; furniture pixels are opaque camera RGB.
- Display: `FurnitureFitOverlayView` draws the cutout over the current GLB room screen.

### GLBRoomActivity inline brain

The primary in-room furniture flow lives in `GLBRoomActivity` (same activity as the WebView room viewer):

The top controls match Swift's room viewer shape: floating back, center ruler/pinch/tap helpers,
recenter/save, and AR resize. The old full-width top band is not used.

1. Tap the **brain** button (bottom-left) to start CameraX analysis on the back camera.
2. **Default mode** auto-segments the highest-confidence primary detection and composites a transparent cutout over the 3D room.
3. Tap the **viewfinder** button while brain is active to enter **full-video** mode:
   - **Identify**: live camera preview + fast detection boxes; tap boxes to pin furniture.
   - **Segment**: camera preview hides; pinned items are segmented with alpha-transparent backgrounds so the **3D room shows through** for fitment checks.
4. Tap **Stop** on the green Segment pill to return to Identify, or tap brain again to exit.

Implementation notes:

- `PreviewView` is bound only in full-video **Identify** mode.
- During **Segment**, CameraX keeps `ImageAnalysis` running but unbinds preview so mask alpha reveals the WebView room (matches Swift `shouldShowLiveCameraPreview`).
- Multi-select segmentation uses mask affinity via `segmentSelectedInstancesAsync`; identify mode stays boxes-only for speed.
- Flat photo rooms use front-facing camera framing in the WebView when depth is under 5 cm.

`FurnitureFitActivity` / `FurnitureFitFragment` remain the dedicated AR room-background path launched from the **AR** pill in the top bar.

## 3D Viewer

Android room outputs are GLB files. `GLBRoomActivity` displays saved and preview rooms with the WebView/Three.js viewer and hosts the furniture segmentation overlay in the same screen.

Bundled sample room assets live in `app/src/main/assets/bundled_rooms/`.

## Local Asset Notes

The root repository ignores `*.onnx` by default. Android's `.gitignore` explicitly allows the tracked Depth Anything ONNX asset under `room_generation/`.

Do not add old native room-generation model exports or large local experiment folders back into app assets.
