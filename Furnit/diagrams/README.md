# Furnit — iOS diagrams

Real SVG flow diagrams for the iOS (Swift) app. Open in any browser, Xcode, or a Markdown/SVG viewer.

| Diagram | Flow |
|---|---|
| [`room-generation-flow.svg`](room-generation-flow.svg) | **Photo → 3D** — home toolbar → `SinglePhotoRoomView` collects a camera/gallery image and sidecar metadata, `DepthAnythingRoomReconstructor` runs GeoCalib + Depth Anything + RTMDet object-anchor work, room measurement produces W×H×D, then the app exports a textured USDZ and opens a room viewer with top measurement/gesture controls and inline brain segmentation. |
| [`rtmdet-swift-flow.svg`](rtmdet-swift-flow.svg) | RTMDet in room viewers — top controls expose ruler/pinch/tap helpers, brain starts `segmentPrimary` (auto primary cutout), and **text.viewfinder** toggles full-video `identifyOnly` (live preview + cluster boxes) and `segmentSelected` (transparent cutout over 3D room). Core ML image input → raw heads → NMS → mask affinity → pixel-union cutout → overlay gestures. |

Legend: green = ANE/GPU (accelerated), blue = CPU, pink = UI/mode control, gray = camera/data, purple = file/web, orange = display. ★ = accelerated stage, ☆ = scalar CPU stage.

Written iOS docs live in [`Furnit/docs/`](../docs/).
