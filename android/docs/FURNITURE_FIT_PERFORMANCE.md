# Furniture Fit frame performance

Android follows the same live-frame ownership rules as Swift even though the model runtimes differ
(FP16 LiteRT GPU with ONNX Runtime fallback on Android, Core ML on iOS).

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

## Reused storage

While a room viewer owns RTMDet, the serial inference executor owns reusable storage for:

- the scaled model-input bitmap;
- model-input ARGB pixels;
- full-frame source and cutout pixel arrays;
- LiteRT NHWC BGR input and named NHWC/NCHW output arrays, or ONNX fallback input storage/tensor;
- copied ARCore Y/U/V planes and direct ARGB output;
- CameraX RGBA packing and periodic color sampling.

Do not make these shared buffers accessible to concurrent inference. If inference becomes parallel,
replace them with a bounded buffer pool first.

## Verification

Run the automated checks:

```bash
./gradlew :app:testDebugUnitTest :app:lintDebug :app:assembleDebugAndroidTest :app:assembleDebug
```

On a physical arm64 device, test portrait and landscape rooms in default segmentation, full-video
Identify, and selected Segment modes. Confirm camera colors, overlay orientation, tap selection,
covered-lens handling, AR measurements, and exiting/re-entering Furniture Fit.

For a debug-signed build, timing lines are emitted by `FurnitureFitManager`:

```bash
adb logcat -s RTMDetLiteRt:I FurnitureFitManager:I tflite:I FurnitureFitAR:I \
  FurnitureFitThermal:D GLBRoomInlineBrainThermal:D -v time
```

Use `stageMillis` to separate preprocessing, LiteRT/ONNX inference, parsing, and mask composition. Measure
UI rendering separately with `adb shell dumpsys gfxinfo com.paafekt.android framestats`; a release
Play build intentionally does not emit application diagnostic logs.

The 2026-08-04 physical-device run confirmed OpenCL with one LiteRT GPU delegate kernel. Before the
final bulk input-buffer copy, steady mask frames were 294–361 ms total versus 1.65–1.87 seconds on
the previous ONNX XNNPACK path. The first installation compiled the GPU program in 28,918 ms and
wrote a serialization artifact under `code_cache`; treat that as cold initialization, not per-frame
latency, and verify cache reuse on a second killed-process launch.
