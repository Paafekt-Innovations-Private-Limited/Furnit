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
- `SinglePhotoRoomReconstructor` writes the current textured GLB room output.
- `RoomGenerationAssets` records the packaged Swift-parity asset contract.
- Generated preview folders are promoted into `files/rooms/` only when the user saves the room.

Android currently packages the Depth Anything metric depth ONNX asset and the existing RTMDet furniture segmentation ONNX asset. GeoCalib needs an Android export before the full Swift metric reconstruction path can be wired.

## Furniture Segmentation

- Model: `rtmdet-ins-m-raw.onnx` in app assets.
- Runtime: ONNX Runtime Android.
- Manager: `app/src/main/java/com/furnit/android/services/FurnitureFitManager.kt`.
- Output: `SegmentationResult.mask` is an ARGB bitmap. Non-furniture pixels have alpha `0`; furniture pixels are opaque camera RGB.
- Display: `FurnitureFitOverlayView` draws the cutout over the current GLB room screen.

## 3D Viewer

Android room outputs are GLB files. `GLBRoomActivity` displays saved and preview rooms with the WebView/Three.js viewer and hosts the furniture segmentation overlay in the same screen.

Bundled sample room assets live in `app/src/main/assets/bundled_rooms/`.

## Local Asset Notes

The root repository ignores `*.onnx` by default. Android's `.gitignore` explicitly allows the tracked Depth Anything ONNX asset under `room_generation/`.

Do not add old native room-generation model exports or large local experiment folders back into app assets.
