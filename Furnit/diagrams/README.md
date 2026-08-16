# Furnit — iOS diagrams

Real SVG flow diagrams for the iOS (Swift) app. Open in any browser, Xcode, or a Markdown/SVG viewer.

| Diagram | Flow |
|---|---|
| [`room-generation-flow.svg`](room-generation-flow.svg) | **Photo → 3D** — home toolbar → `SinglePhotoRoomView` → GeoCalib + Depth Anything + RTMDet run once → measured version-5 projective USDZ → `DepthAnythingPreviewRoomView` opens that final artifact → Save promotes it without reconstruction. Alternate orange **manual** path uses `SinglePhotoRoomReconstructor` + `SyntheticDepthEstimator` → `MeshRoomView`. |
| [`rtmdet-swift-flow.svg`](rtmdet-swift-flow.svg) | RTMDet in room viewers — top controls expose ruler/pinch/tap helpers, brain starts `segmentPrimary` (auto primary cutout), and **text.viewfinder** toggles full-video `identifyOnly` (live preview + cluster boxes) and `segmentSelected` (transparent cutout over 3D room). ODR-delivered RTMDet-Ins-m Core ML model → raw heads → NMS → mask affinity → pixel-union cutout → overlay gestures. |

Legend: green = Metal GPU (accelerated), blue = Swift CPU, pink = UI/mode control, gray = camera/data, purple = file/web, orange = display. ★ = accelerated stage, ☆ = scalar CPU stage.

Written iOS docs live in [`Furnit/docs/`](../docs/).
