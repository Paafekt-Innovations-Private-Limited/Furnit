Furnit Android Skeleton

This folder contains a minimal Android project scaffold that mirrors the iOS app structure found in the repository. It is a starting point — AR and ML functionality are placeholders and require model conversion and further implementation.

Quick start

1. Open the project in Android Studio: open the `android` folder as a project.
2. Let Android Studio sync Gradle and install required SDKs (minSdk 24, compileSdk 34).
3. Build and run on a device (AR features require a real device with ARCore).

Notes

- CoreML models in the iOS app must be converted to TensorFlow Lite or another Android-compatible runtime. See TensorFlow Lite conversion docs.
- AR features currently show a placeholder; integrate ARCore / Sceneform or Google Filament for 3D placement and rendering.
- Services/U2NetSegmentationManager.kt is a stub demonstrating where to put on-device inference logic.

ONNX Runtime (recommended)

- This project includes an exported ONNX model at `android/yoloe-11l-seg-pf.onnx` (if present).
- Preferred approach: use ONNX Runtime Mobile on Android to run the ONNX model directly and avoid fragile ONNX->TFLite conversion steps.
- Place your ONNX model file in `app/src/main/assets/` (or update the asset name in `SmartyPantsManager.initializeOnnx`).
- After placing the ONNX file, open the project in Android Studio and sync Gradle; the `com.microsoft.onnxruntime:onnxruntime-android` dependency is already added.

NOTE: TFLite conversion is paused — the repository's ONNX->TFLite GitHub Actions workflow produced failures and has been disabled. The disabled workflow file is `.github/workflows/convert-onnx-to-tflite.yml.disabled`.
If you later want to retry conversion, re-enable the workflow (rename to `convert-onnx-to-tflite.yml`) or run the conversion scripts in `scripts/` locally or in a container.


3D Model Viewer (SceneView)

The app uses SceneView for 3D model rendering. Models must be in GLB/GLTF format.

**Converting USDZ to GLB:**

iOS uses USDZ format, Android needs GLB. To convert:

1. **Blender (Recommended)**: Install Blender 3.0+, run `python scripts/convert_usdz_to_glb.py`
2. **Reality Converter (macOS)**: Download from Apple Developer, open USDZ, export as GLB
3. **Online**: https://products.aspose.app/3d/conversion/usdz-to-glb

Place GLB files in `app/src/main/assets/models/`:
- `vintage.glb` (from vintage_living_room.usdz)
- `cozy_room.glb` (from cozy_living_room_baked.usdz)

Next steps I can take

- Port UI screens (`ContentView`, `ModelViewerView`, `SettingsView`) to Kotlin/Jetpack Compose.

If you want, I can proceed with any of these next steps now.

Usage example (SmartyPants runtime)

Add the ONNX model to `app/src/main/assets/` (see `scripts/copy_model_to_assets.sh`). In your camera fragment or activity create and initialize the manager:

```kotlin
// create and initialize
val manager = SmartyPantsManager(requireContext())
manager.initializeOnnx("yoloe-11l-seg-pf.onnx")

// when receiving camera frames as Bitmap
manager.segmentImageAsync(frameBitmap) { maskBitmap ->
	runOnUiThread {
		// maskBitmap may be null if inference or post-processing failed
		overlayView.setMask(maskBitmap)
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

`overlayView` is a simple view that draws the mask bitmap above the camera preview (see `SmartyPantsOverlayView` in the project).
