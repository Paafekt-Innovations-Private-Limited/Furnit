# Furnit — iOS diagrams

Real SVG flow diagrams for the iOS (Swift) app. Open in any browser, Xcode, or a Markdown/SVG viewer.

| Diagram | Flow |
|---|---|
| [`room-generation-flow.svg`](room-generation-flow.svg) | **Photo → 3D (two-phase)** — home toolbar → `SinglePhotoRoomView` → **Phase 1:** `PreviewFast` opens `DepthAnythingPreviewRoomView` instantly (no ML; placeholder W×H×D) → **Phase 2 (first save):** `DepthAnythingRoomReconstructor.reconstructWithResult` runs GeoCalib + Depth Anything + RTMDet in parallel, measures W×H×D, exports textured USDZ → saved room in `ModelViewerView`. Alternate orange **manual** path uses `SinglePhotoRoomReconstructor` + `SyntheticDepthEstimator` → `MeshRoomView`. |
| [`rtmdet-swift-flow.svg`](rtmdet-swift-flow.svg) | RTMDet in room viewers — top controls expose ruler/pinch/tap helpers, brain starts `segmentPrimary` (auto primary cutout), and **text.viewfinder** toggles full-video `identifyOnly` (live preview + cluster boxes) and `segmentSelected` (transparent cutout over 3D room). Android-equivalent FP16 TFLite Metal variant → dedicated runtime thread → mandatory audited LiteRT Metal delegate (`cpuNodes=0`) → persistent NHWC-to-NCHW output conversion → raw heads → NMS → mask affinity → pixel-union cutout → overlay gestures. |

Legend: green = Metal GPU (accelerated), blue = Swift CPU, pink = UI/mode control, gray = camera/data, purple = file/web, orange = display. ★ = accelerated stage, ☆ = scalar CPU stage.

Written iOS docs live in [`Furnit/docs/`](../docs/).
