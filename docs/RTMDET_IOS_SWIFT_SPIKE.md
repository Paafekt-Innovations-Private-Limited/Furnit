# RTMDet iOS Swift Spike

This repo contains the active Swift/Core ML path for `RTMDet-Ins-m` in the iOS app.

## What is already wired

- Swift-side RTMDet loader:
  - `Furnit/Services/OnDevice/RTMDetModelService.swift`
- Swift-side RTMDet image/live inference adapter:
  - `Furnit/Services/OnDevice/RTMDetImageInference.swift`
- Live Furniture Fit overlay:
  - `Furnit/Views/FurnitureFit/FurnitureFitView.swift`
- Settings still-image diagnostic:
  - `Furnit/Views/SettingsFurnitureFitImageScanView.swift`

The live room segmentation flow now uses RTMDet raw heads for the Furniture Fit "brain" path. Older
segmentation docs/scripts may remain for comparison, but the Swift active path is RTMDet.

## What is still external

The actual `RTMDet-Ins-m` Core ML model package is not in this repo.

You still need to add one of these bundled model names to the iOS target:

- `rtmdet-ins-m.mlpackage`
- `rtmdet-ins-m.mlmodelc`
- `rtmdet_ins_m.mlpackage`
- `rtmdet_ins_m.mlmodelc`
- `rtmdet-ins-m-coreml.mlpackage`
- `rtmdet-ins-m-coreml.mlmodelc`

Recommended in-repo location:

- `Furnit/Models/RTMDet/`

Helper:

- `scripts/install_rtmdet_ios_model.sh /path/to/rtmdet-ins-m.mlpackage`

The service tries the configured model with a compute-unit fallback chain; test helpers usually force `.cpuOnly` for deterministic host-app unit tests.

1. `.all`
2. `.cpuAndNeuralEngine`
3. `.cpuAndGPU`
4. `.cpuOnly`

## How to test in the iOS app

### Settings still-image scan

1. Add the RTMDet Core ML package to the `Furnit` target.
   Preferred location: `Furnit/Models/RTMDet/`
2. Launch the app.
3. Open `Settings` → `Image scan`.
4. Pick a furniture photo.

What you should see:

- detection boxes over the image
- fused instance-mask cutouts from `RTMDetImageInference`
- a merged mask overlay built by pixel-level RGBA union

The Settings image scan intentionally mirrors the RTMDet live path:

- `maxDetectionCount: nil` (no artificial cap)
- fused `instanceMaskImages`
- no bbox-overlap-only clustering
- no `UIGraphicsImageRenderer` mask blending

### Room viewer full-video smoke test

1. Home → **Photo → 3D** → pick photo (preview opens instantly) or open a **saved** AI room → tap **Save** on preview if testing first-save ML.
2. Tap **brain** (bottom-left). Default `segmentPrimary` should auto-segment one primary item over the 3D room.
3. Tap **text.viewfinder** while brain is active. Live camera preview + cluster boxes should appear.
4. Tap two or more clusters, then tap **Segment**. Preview should hide; transparent cutouts should show over the 3D room.
5. Tap **Stop** to return to live boxes, or brain again to exit.

Relevant room viewers: `ModelViewerView.swift`, `GLBRoomView.swift`, `MeshRoomView.swift`, `SplatRoomView.swift`.
Console filters: `BRAIN FLOW`, `FurnitureFit`, `RTMDet`.

## Current Swift postprocess behavior

- Raw outputs expected: `cls_80/40/20`, `bbox_80/40/20`, `kernel_80/40/20`, and `mask_feat`.
- Decoding chooses the best class per grid cell and applies the configured confidence threshold.
- NMS is class-aware and confidence-first. Area is only a tie-breaker.
- `mask_feat` is copied into a cache-friendly `RawMaskFeatureMatrix`.
- Each selected dynamic kernel builds a raw mask plane.
- `RTMDetMaskAffinityGraph` groups object pieces by mask-level affinity; grouping is class-agnostic.
- Cached selected-mask rebuilds reuse raw outputs and expand selected indices through the same affinity graph.

## Main-flow overlay gestures

The room viewer beneath Furniture Fit also owns pinch zoom. When a segmented cutout is visible,
`FurnitureFitContainerView` must keep two-finger touches so pinch scales the segmented cluster
(`userPinchScale`) rather than zooming the USDZ / GLB / saved PLY room underneath.

Relevant code:

- `FurnitureFitContainerView.handlePinch(_:)`
- `FurnitureFitContainerView.hitTest(_:with:)`
- `FurnitureFitOverlayScaling.resolvedTransform(...)`

## Local output inspection

The current local mask verification script is:

- `scripts/verify_rtmdet_coreml_masks.py`

Example:

```bash
python3 scripts/verify_rtmdet_coreml_masks.py \
  --model Furnit/Models/RTMDet/rtmdet-ins-m.mlpackage \
  --images FurnitTests/rtmdet_repeated_chair_frame.jpg
```

This script runs the raw output contract, decodes boxes/kernels, executes the CondInst mask MLP,
and fails when the selected chair mask is empty or collapsed.

Use this for export inspection only. The Swift app/test path is the source of truth for current image-input behavior, cache behavior, and overlay compositing.

## Current limits

- Python probes may not match the current in-graph image preprocessing path.
- A screenshot of the live overlay is not equivalent to the original camera frame; re-scanning the screenshot can legitimately produce different scores/detections.
- License review still needs to cover the exact checkpoint you export from, not just the OpenMMLab code license.
