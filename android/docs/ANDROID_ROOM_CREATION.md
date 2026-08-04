# Android Room Creation

## Flow

1. `SinglePhotoRoomActivity` lets the user take a room photo or choose one from the library.
2. Selected photos are sampled and EXIF-normalized off the UI thread so full-resolution decode does not block the picker.
3. The user chooses AI generation or manual setup. The method picker is shown immediately; AI generation is posted after the UI frame renders.
4. AI generation starts `PhotoRoomGenerationService.startGenerationInBackground`.
5. `SinglePhotoRoomActivity` handles orientation and screen-size configuration changes without recreation, keeping the selected photo, progress overlay, and in-flight generation callback attached when the phone rotates.
6. The service calls `SinglePhotoRoomReconstructor` with `flatPhotoMesh = true` and no artificial wait time.
7. `GlbGenerator.generateFlatPhotoGlb` writes a single full-photo textured plane GLB (Swift parity) with an embedded JPEG texture.
8. Metadata is written beside the GLB, including room dimensions, depth, photo orientation, and preview state.
9. Preview rooms are saved permanently through `PhotoRoomGenerationService.promoteToLibrary`.
10. Saved rooms open through `GLBRoomActivity`, which receives the saved dimensions for camera framing and ruler display.

Manual setup still uses boundary-based texture crops and `GlbGenerator.generateGlb` (five-plane cuboid).

## Why Flat Photo GLB For AI

The old AI fallback stretched cropped floor/ceiling/wall textures onto cuboid planes, which produced visible **dragged pixels** on the front wall. The flat mesh keeps the entire photo as one texture with clamp-to-edge sampling and unlit materials, matching the Swift single-photo preview.

`GLBRoomActivity` detects thin flat meshes (`roomDepth < 0.05`) and frames the camera in front of the photo plane instead of inside a cuboid.

The room viewer top controls mirror Swift: floating back, center ruler/pinch/tap helpers, recenter/save,
and AR resize. The Android viewer no longer uses a full-width top band.

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

Model assets live in install-time Play Asset Delivery packs (merged into the app's
`AssetManager` at runtime, so load paths are unchanged):

```text
room_generation_models/src/main/assets/room_generation/depth_anything/depth_anything_v2_metric_indoor_small.onnx
room_generation_models/src/main/assets/room_generation/geocalib/geocalib_pinhole_cnn.onnx
rtmdet_models/src/main/assets/rtmdet-ins-m-raw-fp16.tflite
```

The metric measurement pipeline is wired and runs Depth Anything plus RTMDet today. If the GeoCalib ONNX is absent, it uses fallback focal/gravity calibration.

## Runtime

The Android app uses ONNX Runtime for packaged ONNX assets. Room creation no longer depends on native CMake room-generation libraries, separate CPU/GPU variants, or side-loaded model partitions.

## Verification

Use:

```bash
./gradlew :app:assembleDebug
```

After starting AI generation from both standard and wide-angle photos, rotate the phone between
portrait and landscape while the progress overlay is visible. Confirm that the overlay remains,
progress continues, and the generated room opens. Also confirm that the picker appears before
generation work, the preview shows a flat photo wall (not visibly stretched plane crops),
`GLBRoomActivity` recenters the camera in front of the mesh, and the top controls are floating rather
than a full-width band.
