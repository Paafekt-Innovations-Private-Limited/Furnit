# Furnit Android

This folder contains the Android app for Furnit: room capture/reconstruction, GLB/PLY room viewing, and on-device furniture segmentation/cutout.

Quick start

1. Open the project in Android Studio: open the `android` folder as a project.
2. Let Android Studio sync Gradle and install required SDKs (minSdk 24, compileSdk 34).
3. Build and run on a device (AR features require a real device with ARCore).

Notes

- SHARP room reconstruction uses ONNX/ExecuTorch-backed model paths depending on the selected build/runtime.
- Furniture cutout uses RTMDet-Ins raw ONNX through ONNX Runtime in `FurnitureFitManager`.
- GLB rooms render in `GLBRoomActivity` with a WebView/Three.js viewer. The brain button runs segmentation in the same activity and overlays only the transparent cutout; it does not launch a second room-rendering activity.
- PLY/SHARP rooms render in `SharpRoomActivity`; brain segmentation also overlays the existing room view in that activity.

RTMDet / furniture segmentation

- Model: `rtmdet-ins-m-raw.onnx` in app assets.
- Runtime: ONNX Runtime Android.
- Manager: `app/src/main/java/com/furnit/android/services/FurnitureFitManager.kt`.
- Output: `SegmentationResult.mask` is an ARGB bitmap. Non-furniture pixels must have alpha `0`; only furniture pixels are opaque camera RGB.
- Display: `FurnitureFitOverlayView` draws the cutout over the current room screen.

NOTE: TFLite conversion is paused — the repository's ONNX->TFLite GitHub Actions workflow produced failures and has been disabled. The disabled workflow file is `.github/workflows/convert-onnx-to-tflite.yml.disabled`.
If you later want to retry conversion, re-enable the workflow (rename to `convert-onnx-to-tflite.yml`) or run the conversion scripts in `scripts/` locally or in a container.


3D Model Viewer (SceneView)

The app uses SceneView for 3D model rendering. Models must be in GLB/GLTF format.

**Converting USDZ to GLB:**

iOS uses USDZ format, Android needs GLB. To convert:

1. **Blender (Recommended)**: Install Blender 3.0+, run `python scripts/convert_usdz_to_glb.py`
2. **Reality Converter (macOS)**: Download from Apple Developer, open USDZ, export as GLB
3. **Online**: https://products.aspose.app/3d/conversion/usdz-to-glb

Place GLB files in `app/src/main/assets/bundled_rooms/`:
- `vintage.glb` (from vintage_living_room.usdz)
- `cozy_room.glb` (from cozy_living_room_baked.usdz)

Next steps I can take

- Port UI screens (`ContentView`, `ModelViewerView`, `SettingsView`) to Kotlin/Jetpack Compose.

If you want, I can proceed with any of these next steps now.

Usage example (RTMDet runtime)

Add `rtmdet-ins-m-raw.onnx` to `app/src/main/assets/`. In a camera owner activity/fragment, create and initialize the manager:

```kotlin
// create and initialize
val manager = FurnitureFitManager(requireContext())
manager.initializeAuto()

// when receiving camera frames as Bitmap
manager.segmentWithDetectionsAsync(frameBitmap) { result ->
	runOnUiThread {
		// result?.mask may be null if no furniture is detected
		overlayView.setMaskAndDetections(result?.mask, result?.detections ?: emptyList())
	}
}
```

Quick local setup

 - Copy the ONNX model into the app assets (script included):

```bash
./scripts/copy_model_to_assets.sh
```

 - Open the `android` folder in Android Studio and let it sync Gradle. If you prefer a terminal build, ensure you have the Android SDK and Gradle installed, then run (from the `android` directory):

```bash
# using Gradle wrapper if present
./gradlew assembleDebug

# or with system gradle
gradle assembleDebug
```

Notes: this environment does not include the Android SDK or Gradle, so builds must be run locally or in CI with Android tooling available.

`FurnitureFitOverlayView` draws the transparent cutout over the current room view. In GLB brain mode the camera is analysis-only; the camera preview is not displayed as a background.
