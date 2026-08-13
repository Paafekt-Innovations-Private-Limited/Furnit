# Furnit Android

Android app for Furnit room creation, GLB room viewing, and on-device furniture segmentation.

Repository context: [`../docs/READ_FIRST.md`](../docs/READ_FIRST.md). Android doc index:
[`docs/README.md`](docs/README.md). Phone Auth, Play signing, locale country defaults,
and OTP autofill: [`docs/AUTHENTICATION.md`](docs/AUTHENTICATION.md). Supported app
languages, RTL behavior, and catalog verification: [`docs/LOCALIZATION.md`](docs/LOCALIZATION.md).

## Current Room Creation Flow

- Entry point: `SinglePhotoRoomActivity`.
- User input: take a photo or select one from the photo library.
- Picker performance: after image selection, the AI/manual method picker is shown before heavy work starts.
- Decode performance: room photos are sampled and EXIF-normalized off the UI thread instead of decoding full-resolution bitmaps on the main thread.
- AI path: `PhotoRoomGenerationService` writes a **flat full-photo** `room.glb` preview; Save runs metric generation and replaces it with a calibrated projective depth-surface GLB.
- Manual path: the user adjusts room boundaries and exports a textured five-plane GLB room.
- Viewer path: generated rooms open in `GLBRoomActivity`; saved rooms are listed by `ModelManager`.
- GLB performance: AI flat-photo previews embed the texture as JPEG to reduce file size and preview/save latency.

The old native Gaussian-splat stack has been removed from the Android app. There are no separate room-generation build variants and no native room-generation model runtime in the active path.

## GLB Room Viewer And Furniture Fit

`GLBRoomActivity` hosts the WebView/Three.js room and an inline **brain** segmentation overlay:

The room top bar matches the Swift viewer structure: separate floating controls for back, center
ruler/pinch/tap helpers, recenter/save, and AR resize. It is not a full-width band.

Camera navigation is geometry-derived rather than controlled by the Auto Orbit preference. Exterior
views orbit around the room; inside a volumetric room, drag turns the camera in place. Saved
single-photo depth surfaces retain the capture optical center and use field-of-view pinch zoom so
foreground depth cannot separate from its far-photo backing layer. This saved-room interaction is
build-validated; final Android device visual confirmation remains pending as of 2026-08-13.

| Mode | Camera preview | Overlay | Purpose |
|---|---|---|---|
| **Default brain** | Hidden (analysis only) | Auto-segment highest-confidence primary; transparent cutout over 3D room | Quick single-furniture check |
| **Full video — Identify** | Live `PreviewView` | Fast detection boxes; tap to pin furniture | Select one or more items against the live feed |
| **Full video — Segment** | Hidden again | Transparent ARGB cutout(s) over 3D room | Check multi-item fitment in the saved room |

Toggle full-video mode with the in-room **viewfinder** button (`ic_text_viewfinder`) while brain is active. This matches Swift's `text.viewfinder` toolbar control; the old Settings switch has been removed.

RTMDet ownership matches Swift: application launch does not load the model; starting AI photo-room
generation requests it for the upcoming preview; opening a room viewer also ensures its shared
backend asynchronously; activating Fit ensures it again; and leaving a saved-room viewer releases
the backend after accepted inference drains. RTMDet ships as one FP16 LiteRT model: it uses the GPU
delegate on supported devices and otherwise runs the same model through XNNPACK CPU. Identify mode
consumes only cls/bbox results and skips mask-plane/affinity work; segmentation modes use kernels
plus `mask_feat` to build cutouts. The class blacklist is empty, so all 80 COCO classes are eligible.
For segmentation, LiteRT's `mask_feat` remains NHWC and an arm64 NEON mask-head library evaluates
the per-instance dynamic MLP; a Kotlin scalar implementation remains the runtime fallback.

Live frame ownership matches Swift: busy AR frames are dropped before CPU-image/depth work, accepted
AR frames leave the GL thread after a quick plane copy, and CameraX supplies native rotated RGBA with
latest-frame backpressure. Model-input and cutout buffers are reused on the serial inference queue.
See [`docs/FURNITURE_FIT_PERFORMANCE.md`](docs/FURNITURE_FIT_PERFORMANCE.md).

## Packaged Assets

Model assets ship via install-time Play Asset Delivery packs (see `settings.gradle` and
`assetPacks` in `app/build.gradle`), not inside the app module:

- `room_generation_models/src/main/assets/room_generation/depth_anything/depth_anything_v2_metric_indoor_small.onnx`
- `room_generation_models/src/main/assets/room_generation/geocalib/geocalib_pinhole_cnn.onnx`
- `rtmdet_models/src/main/assets/rtmdet-ins-m-raw-fp16.tflite`

Install-time packs are merged into the app's `AssetManager` at runtime, so code still loads
them by the same relative paths (`room_generation/...`, `rtmdet-ins-m-raw-fp16.tflite`).

Do not copy Core ML packages into Android assets.

## Legal Assets And Licenses

Settings → Licenses opens the following app-owned assets without network access:

- `app/src/main/assets/legal/APACHE-2.0.txt` — complete Apache License 2.0 text.
- `app/src/main/assets/legal/LITERT-LICENSE.txt` — LiteRT 1.4.2's complete combined
  license, including its Caffe/BSD attribution.
- `app/src/main/assets/legal/THIRD_PARTY_NOTICES.txt` — runtime/model attribution and
  notices for Paafekt's converted Core ML, ONNX, and LiteRT/TFLite model formats.

`verifyLegalAssets` is wired into `preBuild` in `app/build.gradle`; do not remove or
bypass it. Cross-platform diligence is recorded in
[`../docs/MODEL_LICENSE_AUDIT.md`](../docs/MODEL_LICENSE_AUDIT.md).

## Build

Open the `android` folder in Android Studio, let Gradle sync, then run the `app` configuration on an arm64 device.

Terminal build:

```bash
./gradlew :app:assembleDebug
```

The app module builds one debug variant. Android Studio's Build Variants panel may still be visible because it is part of the IDE, but the project no longer defines the old room-generation variants.

## Useful Logs

```bash
adb logcat -s SinglePhotoRoom:D PhotoRoomGeneration:D RoomGenerationAssets:D GLBRoomActivity:D FurnitureFitManager:D -v time
```

See [`README_ANDROID.md`](README_ANDROID.md) and [`docs/README.md`](docs/README.md) for more detail.
