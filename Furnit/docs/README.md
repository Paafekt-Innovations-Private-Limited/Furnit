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
  anchor → measurement grid → pixel-stable photo-plane USDZ → saved room in
  `ModelViewerView` (or reopen from home).
  Other viewers: `GLBRoomView` (GLB), `MeshRoomView` (manual path), `SplatRoomView` (saved PLY via
  MetalSplatter + SplatIO).
- [`rtmdet-swift-flow.svg`](../diagrams/rtmdet-swift-flow.svg) — RTMDet instance segmentation
  ("brain") in room viewers: top controls expose ruler/pinch/tap helpers, brain default
  auto-segments primary, **text.viewfinder** toggles full-video identify/segment modes, transparent
  cutouts composite over the 3D room, and the Android-equivalent FP16 TFLite graph runs through
  LiteRT's mandatory fully audited Metal delegate (`cpuNodes=0`) → raw-head decode →
  confidence-first NMS → mask affinity → pixel-union cutout.

Room generation (default AI path): **instant preview (no ML)** then **GeoCalib + Depth Anything +
RTMDet object anchor → measured photo-plane USDZ on first save**. Swift entry points: `SinglePhotoRoomViewer.swift`
(`makeDepthAnythingPreviewDestination` for preview; `reconstructWithResult` on save) →
`CameraExifSidecar.swift` → `DepthAnythingRoomReconstructor.swift` → `USDZModel` /
`ModelViewerView`.

Photo orientation follows the displayed pixel dimensions after EXIF rotation, including normalized
`.up` images. The custom 0.5× capture stays landscape on either device side, applies the same
AVFoundation rotation coordination to its preview and encoded still, and keeps the shutter
bottom-center inside the safe area. It prefers Apple's virtual triple/dual-wide camera at its
widest native field of view, requests the largest supported still dimensions and `.quality`
processing, and enables virtual-device fusion so Apple can apply its multi-frame quality/noise
reduction instead of passing a noisier physical ultra-wide frame directly. Standard capture is
unchanged.

The immediate Depth Anything preview remains a flat, pixel-correct photo. New saves preserve that
continuous photo plane instead of depth-displacing its pixels: a single source image has no observed
background behind furniture, so treating estimated depth as a navigable surface creates stretched
edges, holes, and repeated foreground objects. Inferred W×H×D remains authoritative measurement
metadata. The saved viewer uses one-finger pan and field-of-view pinch zoom without rotating the
plane. Projection metadata version 3 marks this contract. Older projective and legacy flat USDZ files
remain readable; the sidecar-based X-aspect correction still applies only to legacy flat files.

Older projective saved rooms keep the authored optical center while pinch changes field of view.
Their drag and D-pad look controls now permit unrestricted yaw and near-vertical pitch without camera
translation. This look-range change is an unconfirmed candidate as of 2026-08-14: it was requested
for push without a compile or final device visual confirmation, so foreground alignment and gray-trace
removal must still be checked manually.

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

`LicensesView` in `Furnit/Views/ContentView.swift` exposes three bundled UTF-8 resources
without requiring network access:

- `Furnit/Licenses/APACHE-2.0.txt` — complete Apache License 2.0 text.
- `Furnit/Licenses/THIRD_PARTY_NOTICES.txt` — shipped model/dataset attribution and
  notices for Paafekt's Core ML, ONNX, and LiteRT/TFLite format conversions.
- `Furnit/Licenses/LITERT-LICENSE.txt` — complete LiteRT/TensorFlow Lite 2.17.0
  combined license and attribution text for the C runtime and Metal delegate.

The notice names the exact Depth Anything V2 Metric Indoor Small variant. All 14 iOS
localizations provide the `licenses.viewNotices` link label. The current diligence and
remaining Hypersim lawyer question live in
[`../../docs/MODEL_LICENSE_AUDIT.md`](../../docs/MODEL_LICENSE_AUDIT.md).

## Docs here
- [`mask-head-accel.md`](mask-head-accel.md) — historical Core ML mask-head optimization
  record; it is not the production LiteRT runtime guide.
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
