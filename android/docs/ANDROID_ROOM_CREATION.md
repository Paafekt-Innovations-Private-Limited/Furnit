# Android Room Creation

## Flow

1. `SinglePhotoRoomActivity` lets the user take a room photo or choose one from the library. Standard capture uses the device camera through `ActivityResultContracts.TakePicture`; the in-app wide capture selects the widest back camera that reports at least a 5 MP JPEG output when one is available, avoiding low-resolution macro/depth sensors.
2. Selected photos are decoded to a quality-bounded 2048 px bitmap and EXIF-normalized off the UI thread. Android 9+ uses `ImageDecoder`; the `BitmapFactory` plus `ExifInterface` path remains for older Android versions and document-provider failures.
3. The user chooses AI generation or manual setup. The method picker is shown immediately; AI generation is posted after the UI frame renders.
4. AI generation starts `PhotoRoomGenerationService.startGenerationInBackground`.
5. `SinglePhotoRoomActivity` handles orientation and screen-size configuration changes without recreation, keeping the selected photo, progress overlay, and in-flight generation callback attached when the phone rotates.
6. The service calls `SinglePhotoRoomReconstructor` with `flatPhotoMesh = true` and no artificial wait time.
7. `GlbGenerator.generateFlatPhotoGlb` writes a single full-photo textured plane GLB for the immediate preview, with an embedded JPEG texture at quality 95.
8. Metadata is written beside the GLB, including room dimensions, depth, photo orientation, and preview state.
9. Save reuses the metric measurement result to pinhole-unproject calibrated depth and writes a projective `photo_room_depth` GLB. Depth-discontinuity triangles are omitted, with a calibrated far-photo backing layer filling those openings.
10. The completed folder is promoted to `files/rooms` transactionally; a failed depth export removes the incomplete saved folder.
11. Saved rooms open through `GLBRoomActivity`, which restores the authored vertical field of view and capture optical center. Pinch changes field of view rather than translating the camera. Drag and D-pad look controls rotate at that fixed optical center with unrestricted yaw and near-vertical pitch, so navigation does not separate foreground depth from the backing photo.

Manual setup still uses boundary-based texture crops and `GlbGenerator.generateGlb` (five-plane cuboid).

## Why The AI Preview Remains Flat

The old AI fallback stretched cropped floor/ceiling/wall textures onto cuboid planes, which produced visible **dragged pixels** on the front wall. The preview keeps the entire photo as one texture with clamp-to-edge sampling and unlit materials, matching the Swift single-photo preview while expensive metric generation remains deferred until Save.

`GLBRoomActivity` detects thin preview meshes (`roomDepth < 0.05`) and frames the camera in front of the photo plane instead of inside a cuboid. Saved `photo_room_depth` meshes use their authored projection metadata. Volumetric/manual rooms switch from exterior orbit to turn-in-place navigation once the camera is inside; the Auto Orbit setting controls only the idle animation.

The expanded saved projective-room look range is an unconfirmed candidate as of 2026-08-14. It was pushed at the user's request without a compile or final device visual confirmation; do not treat the chair/fan alignment and gray-trace issue as resolved until manual testing confirms it.

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
progress continues, and the generated room opens. For wide capture, verify the log reports a
capture-quality back camera and the saved JPEG dimensions are not from a low-resolution auxiliary
sensor. Also confirm that portrait and landscape EXIF inputs remain upright, the picker appears
before generation work, and the preview remains a sharp flat photo without dragged crops or an added
perspective tilt. After Save, reopen the projective room, recenter, pinch in/out, rotate through the
full horizontal range, and look nearly straight up/down; foreground objects must remain aligned and
must not stretch or leave duplicate gray/photo traces. Confirm the
top controls remain floating rather than a full-width band.
