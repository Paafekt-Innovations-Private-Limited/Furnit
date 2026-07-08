# Furnit Android

Android app for Furnit room creation, GLB room viewing, and on-device furniture segmentation.

## Current Room Creation Flow

- Entry point: `SinglePhotoRoomActivity`.
- User input: take a photo or select one from the photo library.
- AI path: `PhotoRoomGenerationService` runs the generic photo-to-room flow and writes a Swift-parity **flat full-photo** `room.glb` preview (no stretched cuboid plane crops).
- Manual path: the user adjusts room boundaries and exports a textured five-plane GLB room.
- Viewer path: generated rooms open in `GLBRoomActivity`; saved rooms are listed by `ModelManager`.

The old native Gaussian-splat stack has been removed from the Android app. There are no separate room-generation build variants and no native room-generation model runtime in the active path.

## GLB Room Viewer And Furniture Fit

`GLBRoomActivity` hosts the WebView/Three.js room and an inline **brain** segmentation overlay:

| Mode | Camera preview | Overlay | Purpose |
|---|---|---|---|
| **Default brain** | Hidden (analysis only) | Auto-segment highest-confidence primary; transparent cutout over 3D room | Quick single-furniture check |
| **Full video — Identify** | Live `PreviewView` | Cluster detection boxes; tap to pin furniture | Select one or more items against the live feed |
| **Full video — Segment** | Hidden again | Transparent ARGB cutout(s) over 3D room | Check multi-item fitment in the saved room |

Toggle full-video mode with the in-room **viewfinder** button (`ic_text_viewfinder`, top-right while brain is active). This matches Swift's `text.viewfinder` toolbar control. A legacy Settings switch still exists but is **not** required for `GLBRoomActivity`.

## Packaged Assets

Room-generation assets are tracked under `app/src/main/assets/room_generation/`:

- `depth_anything/depth_anything_v2_metric_indoor_small.onnx`
- `room_generation_assets.json`
- `geocalib/.gitkeep`

`RoomGenerationAssets` also checks for `rtmdet-ins-m-raw.onnx`, which is packaged at the root of app assets for furniture segmentation.

GeoCalib still needs an Android ONNX or TFLite export before Android can fully match the Swift metric reconstruction pipeline. Do not copy Core ML packages into Android assets.

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
