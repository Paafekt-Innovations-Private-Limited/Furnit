# Furnit — iOS docs

iOS (Swift) app documentation and architecture diagrams.

## Diagrams (`Furnit/diagrams/`)
Real SVG flow diagrams (open in any browser / Xcode preview):

- [`room-generation-flow.svg`](../diagrams/room-generation-flow.svg) — default
  **Photo → 3D** flow: home toolbar → photo capture or library image → camera metadata sidecar →
  GeoCalib + Depth Anything + RTMDet object anchor → measurement grid → textured USDZ →
  room viewer (`ModelViewerView`, `GLBRoomView`, `MeshRoomView`, or saved PLY in `SplatRoomView`)
  with top measurement/gesture controls and inline brain segmentation.
- [`rtmdet-swift-flow.svg`](../diagrams/rtmdet-swift-flow.svg) — RTMDet instance segmentation
  ("brain") in room viewers: top controls expose ruler/pinch/tap helpers, brain default
  auto-segments primary, **text.viewfinder** toggles full-video identify/segment modes, transparent
  cutouts composite over the 3D room, and Core ML raw-head decode → confidence-first NMS →
  mask affinity → pixel-union cutout.

Room generation (default): **GeoCalib + Depth Anything + RTMDet object anchor → USDZ**. The active
Swift path is `SinglePhotoRoomViewer.swift` → `CameraExifSidecar.swift` →
`DepthAnythingRoomReconstructor.swift` → `USDZModel` / room viewer.

## Room viewer smoke test

1. Home → **Photo → 3D** → capture or pick a room photo → AI generation → save/open room.
2. Tap **brain** (bottom-left). Default mode should auto-segment the highest-confidence item over the 3D room.
3. Use the top controls for ruler/pinch/tap guidance, then tap **text.viewfinder** while brain is active. Live camera preview should appear with cluster boxes.
4. Tap two or more furniture clusters.
5. Tap **Segment**. Camera preview should hide; transparent cutouts should composite over the **3D room**.
6. Tap **Stop** to return to live identification boxes, or tap brain again to exit.

Useful Xcode console filter: `FurnitureFit`, `BRAIN FLOW`, `RTMDet`.

## Docs here
- [`mask-head-accel.md`](mask-head-accel.md) — the RTMDet mask-head matmul: problem statement,
  reviewer guidance (profile first, then `cblas_sgemm`), and the per-stage timing instrumentation.
- `Furnit/Views/FurnitureFit/README.md` — RTMDet/FurnitureFit pipeline, room-viewer brain flow,
  Settings scan diagnostic path, mask affinity grouping, pixel-level mask union, and overlay gesture ownership.

## Related iOS docs (repo-root `docs/`)
These are cross-linked with each other (and with `Furnit/Views/FurnitureFit/README.md`) via
`docs/…` relative paths, so they are left in place to avoid breaking those links:

- `docs/IOS_FURNITURE_FIT_ONNX_STYLE_PIPELINE.md`
- `docs/IOS_ROOM_FURNITURE_DIMENSIONS_AND_OVERLAY.md`
- `docs/RTMDET_IOS_SWIFT_SPIKE.md`
- `docs/ON_DEMAND_RESOURCES.md`
- `docs/apple-review-checklist.md`
- `CONTEXT.md` — GeoCalib + Depth Anything pipeline, metric calibration, room-viewer brain/full-video

> To consolidate everything under `Furnit/docs/`, the `docs/…` references inside those files and in
> `Furnit/Views/FurnitureFit/README.md` must be rewritten in the same move.
