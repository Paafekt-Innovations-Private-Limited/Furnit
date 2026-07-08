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

## Room Creation

- `SinglePhotoRoomActivity` owns the take-photo/select-photo screen.
- `PhotoRoomGenerationService` owns the AI room-generation job lifecycle.
- `SinglePhotoRoomReconstructor` writes the textured GLB room output.
- `GlbGenerator.generateFlatPhotoGlb` builds the **AI default**: one full-photo textured plane (Swift parity, avoids dragged/stretched cuboid crops).
- `GlbGenerator.generateGlb` builds the **manual default**: floor, ceiling, and wall planes with cropped photo textures.
- `RoomGenerationAssets` records the packaged Swift-parity asset contract.
- Generated preview folders are promoted into `files/rooms/` only when the user saves the room.

Android currently packages the Depth Anything metric depth ONNX asset and the existing RTMDet furniture segmentation ONNX asset. GeoCalib needs an Android export before the full Swift metric reconstruction path can be wired.

## Furniture Segmentation

- Model: `rtmdet-ins-m-raw.onnx` in app assets.
- Runtime: ONNX Runtime Android.
- Manager: `app/src/main/java/com/furnit/android/services/FurnitureFitManager.kt`.
- Output: `SegmentationResult.mask` is an ARGB bitmap. Non-furniture pixels have alpha `0`; furniture pixels are opaque camera RGB.
- Display: `FurnitureFitOverlayView` draws the cutout over the current GLB room screen.

### GLBRoomActivity inline brain

The primary in-room furniture flow lives in `GLBRoomActivity` (same activity as the WebView room viewer):

1. Tap the **brain** button (bottom-left) to start CameraX analysis on the back camera.
2. **Default mode** auto-segments the highest-confidence primary detection and composites a transparent cutout over the 3D room.
3. Tap the **viewfinder** button (top-right, visible while brain is active) to enter **full-video** mode:
   - **Identify**: live camera preview + cluster bounding boxes; tap boxes to pin furniture.
   - **Segment**: camera preview hides; pinned items are segmented with alpha-transparent backgrounds so the **3D room shows through** for fitment checks.
4. Tap **Stop** on the green Segment pill to return to Identify, or tap brain again to exit.

Implementation notes:

- `PreviewView` is bound only in full-video **Identify** mode.
- During **Segment**, CameraX keeps `ImageAnalysis` running but unbinds preview so mask alpha reveals the WebView room (matches Swift `shouldShowLiveCameraPreview`).
- Multi-select uses cluster union masks via `segmentSelectedInstancesAsync` and affinity-group pins.
- Flat photo rooms use front-facing camera framing in the WebView when depth is under 5 cm.

`FurnitureFitActivity` / `FurnitureFitFragment` remain the dedicated AR room-background path launched from the **AR** pill in the top bar.

## 3D Viewer

Android room outputs are GLB files. `GLBRoomActivity` displays saved and preview rooms with the WebView/Three.js viewer and hosts the furniture segmentation overlay in the same screen.

Bundled sample room assets live in `app/src/main/assets/bundled_rooms/`.

## Local Asset Notes

The root repository ignores `*.onnx` by default. Android's `.gitignore` explicitly allows the tracked Depth Anything ONNX asset under `room_generation/`.

Do not add old native room-generation model exports or large local experiment folders back into app assets.
