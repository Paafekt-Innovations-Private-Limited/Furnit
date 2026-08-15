# Furniture Fit frame performance

Android follows the same live-frame ownership rules as Swift even though the model runtimes differ
(FP16 LiteRT on Android, Core ML on iOS). Android packages one RTMDet model; LiteRT uses the GPU
delegate when supported and otherwise executes that same model with XNNPACK CPU.

`FurnitureFitManager` contains only the active LiteRT RTMDet path; the retired RTMDet ONNX session,
tensor, decoder, and mask fallback were removed. Android still packages ONNX Runtime for the
separate Depth Anything and GeoCalib room-generation models.

The packaged class blacklist is currently empty, so all 80 COCO classes compete during raw-head
decoding on both Android and iOS. Confidence, class-aware NMS, primary scoring, and mask-affinity
grouping still decide which detections become boxes or cutouts.

RTMDet lifecycle also follows Swift: app launch does not load or warm the detector; AI photo-room
generation requests it for the upcoming preview; room-viewer appearance ensures it asynchronously;
Fit activation ensures it again; and saved-viewer disappearance releases the session and reusable
model storage after in-flight inference drains.

## Frame scheduling invariants

- There is at most one frame being converted or handed to inference. Frames never form a FIFO
  queue.
- ARCore frames received while inference is busy are dropped before acquiring the CPU camera image
  or sampling depth. Completion makes the next fresh frame immediately eligible.
- For an accepted ARCore frame, the GL thread only copies the YUV planes. It closes ARCore's image
  immediately; stride-aware, uncompressed YUV-to-ARGB conversion, orientation, and consumer handoff
  run on the camera executor.
- CameraX uses `STRATEGY_KEEP_ONLY_LATEST`, native `RGBA_8888` output, and CameraX output rotation.
  The Java JPEG round trip remains only as a defensive fallback for unexpected YUV output.
- CameraX nominal/fair thermal state permits an inference start every 200 ms and serious permits
  one every 400 ms. ARCore coalesces accepted frames at 280 ms. Severe or higher pauses either path
  while retaining the last overlay.

These rules protect preview rendering. Detection boxes and masks still update at inference cadence;
they are not expected to animate at display refresh rate.

## Full-video segmented overlay controls

After Identify selects an instance and Segment creates the frame-aligned cutout, the cutout uses one
shared draw/hit-test transform. Drag changes its screen-space position and a two-finger pinch changes
its scale from 0.3x through 3x; transparent gaps inside the cutout's opaque-content bounds remain
valid gesture targets. Each newly selected cutout starts at the exact CameraX-aligned 1x pose, while
ordinary incoming segmentation frames preserve the user's transform.

Instrumentation coverage renders a synthetic frame-aligned cutout and verifies that drag moves its
opaque bounds and pinch increases its opaque area. This control change is **device-unconfirmed as of
2026-08-15**; automated build/test results are provisional until portrait and landscape full-video
Segment mode are manually checked on a physical Android device.

## Reused storage

While a room viewer owns RTMDet, the serial inference executor owns reusable storage for:

- the scaled model-input bitmap;
- model-input ARGB pixels;
- full-frame source and cutout pixel arrays;
- LiteRT NHWC BGR input and named output arrays; `mask_feat` stays NHWC for the mask head while
  classification, box, and dynamic-kernel outputs are converted to NCHW for decoding;
- copied ARCore Y/U/V planes and direct ARGB output;
- CameraX RGBA packing and periodic color sampling.

Do not make these shared buffers accessible to concurrent inference. If inference becomes parallel,
replace them with a bounded buffer pool first.

## Native mask head

Segmentation evaluates each detection's 169-coefficient dynamic mask MLP across a 160×160 feature
plane. The arm64 build packages `libfurnit_rtmdet.so`, whose NEON kernel reads LiteRT's native NHWC
`mask_feat` layout and computes the eight hidden channels in two four-lane vectors. If the library
cannot load or rejects its inputs, `RTMDetMaskHead` falls back to the Kotlin scalar implementation;
detection and cutout behavior therefore do not depend on native availability.

JVM tests compare NHWC and legacy NCHW layouts against the previous formulation. The arm64
instrumentation test checks native packaging, scalar parity, repeated-call stability, and a native
speedup. Keep the app's arm64-only split and CMake configuration aligned when changing this path.

## Verification

Run the automated checks:

```bash
./gradlew :app:testDebugUnitTest :app:lintDebug :app:assembleDebugAndroidTest :app:assembleDebug
```

Run `RTMDetMaskHeadNativeTest` on a physical arm64 device using Android Studio or another
asset-pack-preserving deployment. Do not use `app:installDebug`; it strips the install-time model
splits from the host app.

On a physical arm64 device, test portrait and landscape rooms in default segmentation, full-video
Identify, and selected Segment modes. Confirm camera colors, overlay orientation, tap selection,
drag/pinch minimize and maximize, covered-lens handling, AR measurements, and exiting/re-entering
Furniture Fit.

For a debug-signed build, timing lines are emitted by `FurnitureFitManager`:

```bash
adb logcat -s RTMDetLiteRt:I FurnitureFitManager:I tflite:I FurnitureFitAR:I \
  FurnitureFitThermal:D GLBRoomInlineBrainThermal:D -v time
```

Use `stageMillis` to separate preprocessing, LiteRT inference, parsing, and mask composition. Measure
UI rendering separately with `adb shell dumpsys gfxinfo com.paafekt.android framestats`; a release
Play build intentionally does not emit application diagnostic logs.

The 2026-08-04 physical-device run confirmed OpenCL with one LiteRT GPU delegate kernel. Before the
final bulk input-buffer copy, steady mask frames were 294–361 ms total versus 1.65–1.87 seconds on
the previous ONNX XNNPACK path. The first installation compiled the GPU program in 28,918 ms and
wrote a serialization artifact under `code_cache`; treat that as cold initialization, not per-frame
latency, and verify cache reuse on a second killed-process launch.

The 2026-08-13 Pixel 9a run loaded the packaged NEON mask head. Steady raw mask-plane affinity work
for three to six retained detections took 7–19 ms, cutout construction took 27–35 ms, and complete
raw RTMDet segmentation frames took 186–245 ms. The first process after an APK update again spent
28,864 ms creating the GPU backend (189 ms warm-up) and wrote a 107,004,968-byte serialization
artifact; that cold delegate compile remains separate from frame latency.
