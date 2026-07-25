# Furnit Android

Android app for Furnit room creation, GLB room viewing, and on-device furniture segmentation.

## Current Room Creation Flow

- Entry point: `SinglePhotoRoomActivity`.
- User input: take a photo or select one from the photo library.
- Picker performance: after image selection, the AI/manual method picker is shown before heavy work starts.
- Decode performance: room photos are sampled and EXIF-normalized off the UI thread instead of decoding full-resolution bitmaps on the main thread.
- AI path: `PhotoRoomGenerationService` runs the generic photo-to-room flow and writes a Swift-parity **flat full-photo** `room.glb` preview (no stretched cuboid plane crops).
- Manual path: the user adjusts room boundaries and exports a textured five-plane GLB room.
- Viewer path: generated rooms open in `GLBRoomActivity`; saved rooms are listed by `ModelManager`.
- GLB performance: AI flat-photo previews embed the texture as JPEG to reduce file size and preview/save latency.

The old native Gaussian-splat stack has been removed from the Android app. There are no separate room-generation build variants and no native room-generation model runtime in the active path.

## GLB Room Viewer And Furniture Fit

`GLBRoomActivity` hosts the WebView/Three.js room and an inline **brain** segmentation overlay:

The room top bar matches the Swift viewer structure: separate floating controls for back, center
ruler/pinch/tap helpers, recenter/save, and AR resize. It is not a full-width band.

| Mode | Camera preview | Overlay | Purpose |
|---|---|---|---|
| **Default brain** | Hidden (analysis only) | Auto-segment highest-confidence primary; transparent cutout over 3D room | Quick single-furniture check |
| **Full video — Identify** | Live `PreviewView` | Fast detection boxes; tap to pin furniture | Select one or more items against the live feed |
| **Full video — Segment** | Hidden again | Transparent ARGB cutout(s) over 3D room | Check multi-item fitment in the saved room |

Toggle full-video mode with the in-room **viewfinder** button (`ic_text_viewfinder`) while brain is active. This matches Swift's `text.viewfinder` toolbar control; the old Settings switch has been removed.

`FurnitureFitManager` uses a process-wide ONNX Runtime backend so room-viewer sessions reuse the
same `OrtEnvironment`, `OrtSession`, session options, and single-thread inference executor. Identify
mode requests cls/bbox outputs only and skips mask-plane/affinity work; segmentation modes request
kernels plus `mask_feat` to build cutouts.

## Packaged Assets

Model assets ship via install-time Play Asset Delivery packs (see `settings.gradle` and
`assetPacks` in `app/build.gradle`), not inside the app module:

- `room_generation_models/src/main/assets/room_generation/depth_anything/depth_anything_v2_metric_indoor_small.onnx`
- `room_generation_models/src/main/assets/room_generation/geocalib/geocalib_pinhole_cnn.onnx`
- `rtmdet_models/src/main/assets/rtmdet-ins-m-raw.onnx`

Install-time packs are merged into the app's `AssetManager` at runtime, so code still loads
them by the same relative paths (`room_generation/...`, `rtmdet-ins-m-raw.onnx`).

Do not copy Core ML packages into Android assets.

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

See `README_ANDROID.md` and `docs/ANDROID_ROOM_CREATION.md` for more detail.
