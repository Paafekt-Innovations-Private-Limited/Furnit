# RTMDet iOS Runtime

> Historical filename: this is now the active production runtime, not a pending spike.

iOS uses the same RTMDet math, signature, and named tensor contract as Android. Its
`rtmdet-ins-m-raw-fp16.tflite` payload is an iOS-specific Metal-compatible FlatBuffer:
the source graph's four `RELU_0_TO_1` clamps are represented by equivalent
`MAXIMUM(x, 0)` + `MINIMUM(x, 1)` pairs. The app runs it with LiteRT/TensorFlow Lite
2.17.0 and a mandatory Metal delegate. There is no RTMDet Core ML or CPU runtime
fallback.

## Active implementation

| Piece | Location |
|---|---|
| LiteRT C interpreter + Metal delegate | `Furnit/Services/OnDevice/RTMDetLiteRuntime.swift` |
| Shared model/ODR lifecycle | `Furnit/Services/OnDevice/RTMDetModelService.swift` |
| Input preparation + raw-head decoder | `Furnit/Services/OnDevice/RTMDetImageInference.swift` |
| Live Furniture Fit | `Furnit/Views/FurnitureFit/FurnitureFitView.swift` |
| Settings still-image scan | `Furnit/Views/SettingsFurnitureFitImageScanView.swift` |
| Room-generation object anchor | `Furnit/Services/RoomReconstruction/DepthAnythingRoomReconstructor.swift` |

The app owns one `RTMDetLiteRuntime` instance. Live Furniture Fit, Settings image
scan, and the first-save object anchor serialize inference through that runtime rather
than loading duplicate model copies.

## Shipped artifacts

- Model: `Furnit/Models/RTMDet/rtmdet-ins-m-raw-fp16.tflite`
- Git LFS SHA-256 of the current model payload:
  `f13a4bf62e79284ae1b2f872c8ab7288767475fc2864af627c7dd79479bf1757`
- Reviewed Android source SHA-256:
  `7edbd6692733d42a70344999aa5815762585c2a785b0e47cead4d786d4fb854d`
- Reproducible rewrite: `scripts/rewrite_rtmdet_ios_metal_graph.py`
- ODR tag: `RTMDetModel`
- Official static runtime frameworks:
  - `Vendor/LiteRT/TensorFlowLiteC.xcframework`
  - `Vendor/LiteRT/TensorFlowLiteCMetal.xcframework`
- Offline license/attribution: `Furnit/Licenses/LITERT-LICENSE.txt`

The former `.mlpackage` exports and Core ML verification scripts remain experimental
history only. The Xcode target excludes the old local RTMDet package, and the loader
does not search for it.

## Tensor contract

- Signature: `serving_default`
- Input: `input`, float32 NHWC `[1, 640, 640, 3]`
- Pixel contract: raw BGR `0...255`; normalization is inside the graph
- Preprocess: stretch to 640×640, matching Android
- Outputs:
  - `cls_80/40/20`
  - `bbox_80/40/20`
  - `kernel_80/40/20`
  - `mask_feat`

LiteRT stores outputs as NHWC. After each delegated invocation, the runtime uses
`TfLiteTensorCopyToBuffer` to extract every output into persistent app-owned storage,
then physically converts it into contiguous NCHW storage exactly as Android does.
Creation, warm-up, inference, output consumption, and teardown stay on one persistent
worker thread. The delegate uses Android-parity precision and quantization options.
After Metal partitioning, a native no-op audit inspects the execution plan; model load
fails unless every executable node belongs to Metal. Automatic delegate fallback is
also disabled.

The rewrite was checked against the Android source with a real RTMDet input: all ten
named CPU-reference output tensors were bit-for-bit equal. A separate runtime guard
also rejects the impossible saturated-class pattern seen when the old graph was only
partly delegated, before mask construction can display a corrupted overlay.

## App smoke checks

### Settings image scan

1. Install on a physical iPhone.
2. Open Settings → Image scan.
3. Pick a furniture photo.
4. Confirm the console contains `verified full LiteRT Metal delegation` with
   `cpuNodes=0` and no unsupported-operation warning.
5. Confirm boxes, fused instance cutouts, and the merged pixel-union mask appear.

The Settings path intentionally matches live RTMDet behavior: uncapped post-NMS
detections, mask-affinity fusion, and no bbox-only clustering or renderer-based mask
blending.

### Room viewer

1. Open a saved room and tap **brain**.
2. Confirm `segmentPrimary` produces the primary transparent cutout.
3. Tap **text.viewfinder** and confirm live cluster boxes.
4. Select clusters and tap **Segment**; the preview should hide and the cutouts should
   appear over the 3D room.

### First-save room measurement

Create a Photo → 3D room and tap Save. GeoCalib, Depth Anything Metric Indoor Small,
and the same shared RTMDet runtime execute in the save pipeline; RTMDet is not loaded
again as a second interpreter.

## Postprocess behavior

- Decode the raw 80/40/20 heads at the configured confidence floor.
- Apply confidence-first, class-aware NMS.
- Build dynamic-kernel mask planes from `mask_feat`.
- Group object pieces with class-agnostic mask affinity.
- Rebuild selected masks from cached raw data and perform pixel-level RGBA union.

See `Furnit/Views/FurnitureFit/README.md`,
`docs/IOS_FURNITURE_FIT_ONNX_STYLE_PIPELINE.md`, and
`Furnit/diagrams/rtmdet-swift-flow.svg` for the owning documentation.
