# Furnit — iOS docs

iOS (Swift) app documentation and architecture diagrams.

Repository-wide active context: [`../../docs/READ_FIRST.md`](../../docs/READ_FIRST.md).
Architecture/code ownership: [`../../docs/architecture.md`](../../docs/architecture.md)
and [`../../docs/architecture/CODE_MAP.md`](../../docs/architecture/CODE_MAP.md).

## Diagrams (`Furnit/diagrams/`)
Real SVG flow diagrams (open in any browser / Xcode preview):

- [`room-generation-flow.svg`](../diagrams/room-generation-flow.svg) — default
  **Photo → 3D** flow (two-phase): home toolbar → photo capture or library image → camera metadata
  sidecar → **instant preview** (`PreviewFast`, no ML, placeholder dims) in
  `DepthAnythingPreviewRoomView` → **first save** runs GeoCalib + Depth Anything + RTMDet object
  anchor → measurement grid → textured USDZ → saved room in `ModelViewerView` (or reopen from home).
  Other viewers: `GLBRoomView` (GLB), `MeshRoomView` (manual path), `SplatRoomView` (saved PLY via
  MetalSplatter + SplatIO).
- [`rtmdet-swift-flow.svg`](../diagrams/rtmdet-swift-flow.svg) — RTMDet instance segmentation
  ("brain") in room viewers: top controls expose ruler/pinch/tap helpers, brain default
  auto-segments primary, **text.viewfinder** toggles full-video identify/segment modes, transparent
  cutouts composite over the 3D room, and Core ML raw-head decode → confidence-first NMS →
  mask affinity → pixel-union cutout.

Room generation (default AI path): **instant preview (no ML)** then **GeoCalib + Depth Anything +
RTMDet object anchor → USDZ on first save**. Swift entry points: `SinglePhotoRoomViewer.swift`
(`makeDepthAnythingPreviewDestination` for preview; `reconstructWithResult` on save) →
`CameraExifSidecar.swift` → `DepthAnythingRoomReconstructor.swift` → `USDZModel` /
`ModelViewerView`.

## Room viewer smoke test

1. Home → **Photo → 3D** → capture or pick a room photo → AI path opens preview instantly → tap
   **Save** to run metric generation → room appears in home list.
2. Tap **brain** (bottom-left). Default mode should auto-segment the highest-confidence item over the 3D room.
3. Use the top controls for ruler/pinch/tap guidance, then tap **text.viewfinder** while brain is active. Live camera preview should appear with cluster boxes.
4. Tap two or more furniture clusters.
5. Tap **Segment**. Camera preview should hide; transparent cutouts should composite over the **3D room**.
   Drag a cutout to move it and pinch it to resize it.
6. Tap **Stop** to return to live identification boxes, or tap brain again to exit.

Useful Xcode console filter: `FurnitureFit`, `BRAIN FLOW`, `RTMDet`.

## Settings licenses and attributions

`LicensesView` in `Furnit/Views/ContentView.swift` exposes two bundled UTF-8 resources
without requiring network access:

- `Furnit/Licenses/APACHE-2.0.txt` — complete Apache License 2.0 text.
- `Furnit/Licenses/THIRD_PARTY_NOTICES.txt` — shipped model/dataset attribution and
  notices for Paafekt's Core ML, ONNX, and LiteRT/TFLite format conversions.

The notice names the exact Depth Anything V2 Metric Indoor Small variant. All 14 iOS
localizations provide the `licenses.viewNotices` link label. The current diligence and
remaining Hypersim lawyer question live in
[`../../docs/MODEL_LICENSE_AUDIT.md`](../../docs/MODEL_LICENSE_AUDIT.md).

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
- `docs/MODEL_LICENSE_AUDIT.md`
- `docs/READ_FIRST.md` — compact current context and settled cross-platform facts
- `docs/architecture.md` — active architecture entry point

Detailed iOS docs intentionally remain beside the iOS code. Shared docs remain in the
repository-root `docs/` directory and are indexed by `docs/README.md`.
