# Furnit — iOS docs

iOS (Swift) app documentation and architecture diagrams.

Repository-wide active context: [`../../docs/READ_FIRST.md`](../../docs/READ_FIRST.md).
Architecture/code ownership: [`../../docs/architecture.md`](../../docs/architecture.md)
and [`../../docs/architecture/CODE_MAP.md`](../../docs/architecture/CODE_MAP.md).

## Diagrams (`Furnit/diagrams/`)
Real SVG flow diagrams (open in any browser / Xcode preview):

- [`room-generation-flow.svg`](../diagrams/room-generation-flow.svg) — default
  **Photo → 3D** flow: home toolbar → photo capture or library image → camera metadata sidecar →
  GeoCalib + Depth Anything + RTMDet object anchor → measurement grid → version-5 single-surface
  projective USDZ → exact RealityKit preview in `DepthAnythingPreviewRoomView` → Save promotes the
  inspected USDZ to `ModelViewerView`/home without reconstructing it again.
  Other viewers: `GLBRoomView` (GLB), `MeshRoomView` (manual path), `SplatRoomView` (saved PLY via
  MetalSplatter + SplatIO).
- [`rtmdet-swift-flow.svg`](../diagrams/rtmdet-swift-flow.svg) — RTMDet instance segmentation
  ("brain") in room viewers: top controls expose ruler/pinch/tap helpers, brain default
  auto-segments primary, **text.viewfinder** toggles full-video identify/segment modes, transparent
  cutouts composite over the 3D room, and the Android-equivalent FP16 TFLite graph runs through
  LiteRT's mandatory fully audited Metal delegate (`cpuNodes=0`) → raw-head decode →
  confidence-first NMS → mask affinity → pixel-union cutout.

Room generation (default AI path): **GeoCalib + Depth Anything + RTMDet object anchor → measured
single-surface projective USDZ before preview**, followed by an exact RealityKit preview and
byte-preserving Save promotion. Swift entry points: `SinglePhotoRoomViewer.swift`
(`makeDepthAnythingPreviewDestination` generates the preview artifact; Save copies it) →
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

The Depth Anything preview displays the generated final USDZ, so preview and reopened saved rooms
share the same continuous perspective surface. Generation depth-unprojects the displayed pixels
before preview and does not add a second
completed-background enclosure; keeping one opaque surface avoids the prior translucent shell while
limited capture-eye translation and look-around expose the metric depth shape. A single photograph
still contains no true hidden pixels, so off-axis views are reprojections rather than newly observed
content. Inferred W×H×D remains authoritative measurement metadata. Projection metadata version 5
marks this contract. Older projective and legacy flat USDZ files remain readable; the sidecar-based
X-aspect correction still applies only to legacy flat files.

The current preview/save appearance candidate uses a single opaque, unlit texture without interpolated
vertex color, stages `room.jpg` through `SCNSceneExportDelegate`, and normalizes imported photo-room
materials in RealityKit with a neutral black background and no image-based lighting. The camera and
movement envelope use the same capture convention as Android/glTF (forward is negative Z). A generic
iPhoneOS Debug build succeeds, but preview-versus-reopen appearance remains device-unconfirmed as of
2026-08-15. Test a newly generated room through preview → Save → reopen; opening an older room
exercises only viewer-side compatibility and does not rewrite its USDZ.

The Settings **Infinite Zoom** switch defaults off. With it off, the existing photo/room camera
envelopes and viewer-specific zoom limits are unchanged. The 2026-08-16 opt-in candidate propagates
the setting through RealityKit USDZ, immediate photo preview, Metal splat, GLB, and manual-mesh
viewers; enabled viewers use wider clipping/zoom ranges and skip their normal positional boundary
clamp where applicable. Generic and signed iPhoneOS Debug builds succeed, and the saved preference
was verified on a connected device, but repeated iOS manual checks have not confirmed the intended
behavior. Treat this as an unconfirmed candidate until a fresh continuous-depth USDZ and the other
viewer formats are tested on-device with the switch both off and on.

`FurnitUITests/SavedRoomNavigationUITests.swift` adds a create → preview → save → reopen navigation
regression using a programmatically generated synthetic room image. It checks zoom, pan and D-pad
frame changes plus renderer-background exposure. The test target compiles independently of any
private room-photo fixture; the visual test still requires an iOS device/test destination.

## Room viewer smoke test

1. Home → **Photo → 3D** → capture or pick a room photo → wait for metric generation → inspect the
   exact USDZ preview → tap **Save** → reopen it from the home list and compare framing/appearance.
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

In debug builds, the Settings Developer section is additionally restricted to the
authenticated Firebase test number `+1 650-555-3434`. Signing out or switching to a
different identity hides the section and disables any persisted debug mode.

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
