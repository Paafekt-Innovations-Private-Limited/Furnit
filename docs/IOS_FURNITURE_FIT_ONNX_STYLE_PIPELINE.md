# iOS Furniture Fit — RTMDet Core ML Pipeline

This document summarizes the current iOS Furniture Fit segmentation path. The implementation uses
**RTMDet-Ins-m** through Core ML and keeps the Android-style helper name
`FurnitureFitOnnxStylePipeline` only for historical parity with older preprocessing/mask code.
There is no ONNX Runtime dependency in the iOS app.

## Primary Sources

| Piece | Location |
|------|----------|
| Model loading / ODR | `Furnit/Services/OnDevice/RTMDetModelService.swift` |
| Core ML inference + raw-head postprocess | `Furnit/Services/OnDevice/RTMDetImageInference.swift` |
| Live room flow, selection, gestures | `Furnit/Views/FurnitureFit/FurnitureFitView.swift` |
| Android-style mask helpers | `Furnit/Services/OnDevice/FurnitureFitOnnxStylePipeline.swift` |
| Main in-project doc | `Furnit/Views/FurnitureFit/README.md` |
| Diagram | `Furnit/diagrams/rtmdet-swift-flow.svg` |

## Flow

```text
room viewer brain tap OR Settings image scan
  -> frame gate + thermal cadence
  -> AVCapture live feed (preview on in identifyOnly; hidden in full-video segmentSelected)
  -> resize to model input
  -> Core ML image input
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

Toggle full-video with the in-room **text.viewfinder** button (top-right while brain is active).
This matches Android `GLBRoomActivity`'s viewfinder toggle. A legacy Settings switch still exists
but room viewers drive the mode from the in-room button.

## Current Behavior

- RTMDet raw outputs are `cls/bbox/kernel` at 80/40/20 plus `mask_feat`.
- Core ML image input is preferred; the model graph owns its expected normalization.
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
- Frame processing is serial with back-pressure: one inference at a time.
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
