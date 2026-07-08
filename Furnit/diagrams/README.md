# Furnit — iOS diagrams

Real SVG flow diagrams for the iOS (Swift) app. Open in any browser, Xcode, or a Markdown/SVG viewer.

| Diagram | Flow |
|---|---|
| [`rtmdet-swift-flow.svg`](rtmdet-swift-flow.svg) | RTMDet instance segmentation ("brain") live/still loop — `RTMDetImageInference` + `FurnitureFitView` + Settings image scan. Camera/still image → Core ML image input → raw heads → confidence-first NMS → mask affinity → pixel-union cutout → overlay gestures/display. |

**Room generation (default, no diagram yet):** single photo → **GeoCalib** (focal, letterboxed full frame) → **Depth Anything** metric depth (same grid) → chair-anchor depth scale → depth-spread W×H×D → USDZ → `ModelViewerView`. Code: `GeoCalibCalibrationService.swift`, `DepthAnythingRoomReconstructor.swift`, `SinglePhotoRoomViewer.swift`.

| Legacy | Flow |
|---|---|
| [`splat-swift-flow.svg`](splat-swift-flow.svg) | Retired Gaussian-splat path — not used for default iOS room creation. |

Legend: green = ANE/GPU (accelerated), blue = CPU, gray = camera/data, purple = file/web, orange =
display. ★ = accelerated stage, ☆ = scalar CPU stage.
