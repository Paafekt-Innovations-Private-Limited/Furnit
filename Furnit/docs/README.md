# Furnit — iOS docs

iOS (Swift) app documentation and architecture diagrams.

## Diagrams (`Furnit/diagrams/`)
Real SVG flow diagrams (open in any browser / Xcode preview):

- [`rtmdet-swift-flow.svg`](../diagrams/rtmdet-swift-flow.svg) — RTMDet instance-segmentation
  ("brain") live/still loop: camera or Settings image scan → Core ML image input → raw-head decode
  → confidence-first NMS → mask affinity → pixel-union cutout → overlay gestures/display.
- [`sharp-swift-flow.svg`](../diagrams/sharp-swift-flow.svg) — SHARP room reconstruction: single
  photo → Core ML Gaussian-splat → PLY + `.splatcache` → SharpRoomView render → dimensions.

## Docs here
- [`mask-head-accel.md`](mask-head-accel.md) — the RTMDet mask-head matmul: problem statement,
  reviewer guidance (profile first, then `cblas_sgemm`), and the per-stage timing instrumentation.
- `Furnit/Views/FurnitureFit/README.md` — current RTMDet/FurnitureFit live path, Settings scan
  diagnostic path, mask affinity grouping, pixel-level mask union, and overlay gesture ownership.

## Related iOS docs (repo-root `docs/`)
These are cross-linked with each other (and with `Furnit/Views/FurnitureFit/README.md`) via
`docs/…` relative paths, so they are left in place to avoid breaking those links:

- `docs/IOS_FURNITURE_FIT_ONNX_STYLE_PIPELINE.md`
- `docs/IOS_ROOM_FURNITURE_DIMENSIONS_AND_OVERLAY.md`
- `docs/RTMDET_IOS_SWIFT_SPIKE.md`
- `docs/ON_DEMAND_RESOURCES.md`
- `docs/apple-review-checklist.md`
- `docs/SHARP_BLACK_PATCHES_FIX.md`, `docs/SHARP_SINGLE_CLASSIC_PLY.md`

> To consolidate everything under `Furnit/docs/`, the `docs/…` references inside those files and in
> `Furnit/Views/FurnitureFit/README.md` must be rewritten in the same move.
