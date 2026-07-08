# Android Room Creation

## Flow

1. `SinglePhotoRoomActivity` lets the user take a room photo or choose one from the library.
2. The user chooses AI generation or manual setup.
3. AI generation starts `PhotoRoomGenerationService.startGenerationInBackground`.
4. The service calls `SinglePhotoRoomReconstructor` with `flatPhotoMesh = true`.
5. `GlbGenerator.generateFlatPhotoGlb` writes a single full-photo textured plane GLB (Swift parity).
6. Metadata is written beside the GLB, including room dimensions, photo orientation, and preview state.
7. Preview rooms are saved permanently through `PhotoRoomGenerationService.promoteToLibrary`.
8. Saved rooms open through `GLBRoomActivity`.

Manual setup still uses boundary-based texture crops and `GlbGenerator.generateGlb` (five-plane cuboid).

## Why Flat Photo GLB For AI

The old AI fallback stretched cropped floor/ceiling/wall textures onto cuboid planes, which produced visible **dragged pixels** on the front wall. The flat mesh keeps the entire photo as one texture with clamp-to-edge sampling and unlit materials, matching the Swift single-photo preview.

`GLBRoomActivity` detects thin flat meshes (`roomDepth < 0.05`) and frames the camera in front of the photo plane instead of inside a cuboid.

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

After creating an AI room, confirm the preview shows a flat photo wall (not visibly stretched plane crops) and that `GLBRoomActivity` recenters the camera in front of the mesh.
