# iOS Furniture Fit — RTMDet LiteRT Metal Pipeline

This document summarizes the current iOS Furniture Fit segmentation path. The implementation uses
Android's **RTMDet-Ins-m FP16 TFLite math and tensor contract** through an iOS-specific,
Metal-compatible graph variant and LiteRT's iOS Metal delegate. The variant only replaces four
`RELU_0_TO_1` clamps with equivalent max/min pairs. It keeps the Android-style helper name
`FurnitureFitOnnxStylePipeline` only for historical parity with older preprocessing/mask code.
There is no ONNX Runtime dependency and no Core ML/CPU fallback for RTMDet in the iOS app.

## Primary Sources

| Piece | Location |
|------|----------|
| Model loading / ODR | `Furnit/Services/OnDevice/RTMDetModelService.swift` |
| LiteRT C + Metal runtime | `Furnit/Services/OnDevice/RTMDetLiteRuntime.swift` |
| Input + raw-head postprocess | `Furnit/Services/OnDevice/RTMDetImageInference.swift` |
| Live room flow, selection, gestures | `Furnit/Views/FurnitureFit/FurnitureFitView.swift` |
| Android-style mask helpers | `Furnit/Services/OnDevice/FurnitureFitOnnxStylePipeline.swift` |
| Main in-project doc | `Furnit/Views/FurnitureFit/README.md` |
| Diagram | `Furnit/diagrams/rtmdet-swift-flow.svg` |

## Flow

```text
room viewer brain tap OR Settings image scan
  -> frame gate + thermal cadence
  -> AVCapture live feed (preview on in identifyOnly; hidden in full-video segmentSelected)
  -> stretch to 640x640
  -> raw BGR float32 NHWC input
  -> dedicated LiteRT worker + LiteRT 2.17 Metal delegate
  -> execution-plan audit (zero CPU nodes required)
  -> RTMDet raw heads
  -> decode candidates
  -> confidence-first class-aware NMS
  -> mask planes + affinity clusters (union bboxes in full-video)
  -> pixel-level RGBA cutout union
  -> overlay display + gesture ownership over USDZ / GLB / PLY room
```

## Room viewer modes

`GLBRoomView`, `ModelViewerView`, `MeshRoomView`, and `SplatRoomView` host inline Furniture Fit.
Modes map to `FurnitureFitSegmentationMode` in `FurnitureFitOverlaySupport.swift`:

| Mode | Camera preview | Overlay | Purpose |
|---|---|---|---|
| **Brain default** (`segmentPrimary`) | Hidden (analysis only) | Auto-segment highest-confidence primary; transparent cutout over 3D room | Quick single-furniture check |
| **Full video — Identify** (`identifyOnly` + viewfinder) | Live `AVCaptureVideoPreviewLayer` | Cluster detection boxes; tap to pin furniture | Select one or more items against the live feed |
| **Full video — Segment** (`segmentSelected` + viewfinder) | Hidden again | Transparent cutout(s) over 3D room | Check multi-item fitment in the saved room |

Toggle full-video with the in-room **text.viewfinder** button while brain is active. The same top
control model is mirrored by Android `GLBRoomActivity`: floating back, center ruler/pinch/tap
helpers, recenter/save, AR, and the viewfinder toggle for full-video identify.

## Current Behavior

- RTMDet raw outputs are `cls/bbox/kernel` at 80/40/20 plus `mask_feat`.
- One shared `RTMDetLiteRuntime` serves live, still-image, and room-anchor inference.
- The model graph owns normalization; Swift writes raw BGR `0...255` values directly
  into LiteRT's input allocation.
- The checked-in iOS graph is built by `scripts/rewrite_rtmdet_ios_metal_graph.py` from
  the pinned Android source. All ten outputs were bit-identical on the CPU reference
  check before it was installed.
- After each Metal-delegated invocation, LiteRT outputs are extracted with
  `TfLiteTensorCopyToBuffer` into persistent storage and physically converted from
  NHWC to contiguous NCHW exactly as Android does. The delegate uses Android-parity
  precision/quantization options and remains mandatory. Creation, warm-up, inference,
  and destruction stay on one thread; the app rejects any delegated plan with CPU nodes.
- NMS is confidence-first and class-aware. Area is only a deterministic tie-breaker.
- Settings image scan uses the same still-image path as the live room flow: uncapped detections,
  fused instance masks, and pixel-level RGBA union.
- Object-piece fusion uses mask affinity over raw mask planes. The grouping is class-agnostic, so a
  multi-piece physical object can be segmented as one object without hard-coded chair rules.
- Full video mode displays cluster-level union boxes. Tapping a cluster selects all members for
  segmentation. During `segmentSelected`, the camera preview hides so transparent cutouts composite
  over the 3D room underneath.
- When segmented furniture is visible over a USDZ / GLB / saved PLY room, the Furniture Fit overlay
  must claim two-finger touches so pinch scales the furniture cutout, not the room camera.

## Performance / Thermal Notes

- Live identify cadence is about 5 fps (`rtmdetLiveTargetInterval = 200ms`).
- Serious thermal state backs off cadence; critical thermal state pauses inference and keeps the
  last displayed boxes.
- Inference is skipped entirely while independent overlay placement is active.
- Frame processing is serial with back-pressure on the dedicated LiteRT worker: one inference at a time.
- Debug logging surfaces per-stage timings and memory points; see `Furnit/docs/mask-head-accel.md`
  for the mask-head profiling notes.

## Related Reading

- `Furnit/Views/FurnitureFit/README.md` — detailed current behavior, memory fixes, mask affinity,
  multi-select placement, cadence, and gesture ownership.
- `Furnit/diagrams/rtmdet-swift-flow.svg` — visual flow for live/still segmentation.
- `docs/IOS_ROOM_FURNITURE_DIMENSIONS_AND_OVERLAY.md` — Depth Anything USDZ room dimensions,
  pinhole furniture sizing, fitment ratios, and overlay scale.
- `docs/ON_DEMAND_RESOURCES.md` — RTMDet ODR and bundled room-generation models.
- `docs/RTMDET_IOS_SWIFT_SPIKE.md` — raw-head export expectations and Swift postprocess details.
