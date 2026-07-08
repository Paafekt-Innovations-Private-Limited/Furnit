# Android Room Creation

## Flow

1. `SinglePhotoRoomActivity` lets the user take a room photo or choose one from the library.
2. The user chooses AI generation or manual setup.
3. AI generation starts `PhotoRoomGenerationService.startGenerationInBackground`.
4. The service calls `SinglePhotoRoomReconstructor` with default room boundaries and writes `room.glb`.
5. Metadata is written beside the GLB, including room dimensions, photo orientation, and preview state.
6. Preview rooms are saved permanently through `PhotoRoomGenerationService.promoteToLibrary`.
7. Saved rooms open through `GLBRoomActivity`.

## Active Outputs

Generated room folders use GLB output:

```text
files/room_previews/<room-id>/room.glb
files/room_previews/<room-id>/metadata.txt
files/rooms/<room-id>/room.glb
files/rooms/<room-id>/metadata.txt
```

`ModelManager` only loads user-created room folders from `files/rooms`.

## Assets

Tracked Android room-generation assets:

```text
app/src/main/assets/room_generation/depth_anything/depth_anything_v2_metric_indoor_small.onnx
app/src/main/assets/room_generation/room_generation_assets.json
app/src/main/assets/rtmdet-ins-m-raw.onnx
```

Expected but not yet exported for Android:

```text
app/src/main/assets/room_generation/geocalib/geocalib_pinhole_cnn.onnx
```

GeoCalib currently exists in the shared project as Core ML/checkpoint artifacts. Android needs an ONNX or TFLite export before the full Swift metric reconstruction path can be wired.

## Runtime

The Android app uses ONNX Runtime for packaged ONNX assets. Room creation no longer depends on native CMake room-generation libraries, separate CPU/GPU variants, or side-loaded model partitions.

## Verification

Use:

```bash
./gradlew :app:assembleDebug
rg -n "old-room-backend-token" app/src/main/java app/src/main/res app/build.gradle
```
