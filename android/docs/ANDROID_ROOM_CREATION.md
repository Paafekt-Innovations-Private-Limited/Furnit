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
9. Save reruns metric measurement with foreground masks enabled. RTMDet mask unions and calibrated depth discontinuities form a conservative foreground layer; `LayeredDepthRoomCompletion` fills masked background color and inverse depth from nearby known structure.
10. `GlbGenerator.generateDepthPhotoGlb` writes a version-5 `photo_room_depth` GLB with separate completed-background and foreground geometry. The old full-photo far backing plane is not emitted. If a reliable layered asset cannot be generated, Save writes a flat photo GLB instead.
11. The completed folder is promoted to `files/rooms` transactionally; a failed export removes the incomplete saved folder. Metadata records projection version and whether the background is completed.
12. Saved rooms open through `GLBRoomActivity`, which restores the authored field of view and camera-validity envelope. Pinch permits bounded forward/backward translation for version-5 assets; drag and D-pad look are clamped to source-image coverage. Flat fallback and legacy assets retain their compatible navigation paths.

Manual setup still uses boundary-based texture crops and `GlbGenerator.generateGlb` (five-plane cuboid).

## Why The AI Preview Remains Flat

The old AI fallback stretched cropped floor/ceiling/wall textures onto cuboid planes, which produced visible **dragged pixels** on the front wall. The preview keeps the entire photo as one texture with clamp-to-edge sampling and unlit materials, matching the Swift single-photo preview while expensive metric generation remains deferred until Save.

`GLBRoomActivity` detects thin preview meshes (`roomDepth < 0.05`) and frames the camera in front of the photo plane instead of inside a cuboid. Saved `photo_room_depth` meshes use their authored projection and camera-envelope metadata. Volumetric/manual rooms switch from exterior orbit to turn-in-place navigation once the camera is inside; the Auto Orbit setting controls only the idle animation.

The version-5 layered completion/navigation path is an unconfirmed candidate as of 2026-08-14.
Automated build and test-target compilation are provisional; do not treat the chair/fan alignment,
duplicate foreground, or gray-hole issue as resolved until manual device testing confirms it.

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
perspective tilt. After Save, reopen the room and verify whether it is a layered version-5 asset or
the flat fallback. For a layered asset, recenter, pinch within the authored movement envelope, and
drag/D-pad toward each bounded look limit. Foreground objects must remain aligned; newly revealed
background must not repeat foreground pixels, stretch at depth edges, or expose renderer-gray holes.
Confirm the top controls remain floating rather than a full-width band.

`SavedRoomNavigationE2ETest` provides a create → preview → save → reopen regression with a synthetic
image generated at test runtime. It checks navigation frame changes and renderer-gray exposure, but
must be run on a suitable arm64 Android device and does not replace manual judgment on the reported
chair/fan photograph.
