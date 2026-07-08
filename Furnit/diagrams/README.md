# Furnit — iOS diagrams

Real SVG flow diagrams for the iOS (Swift) app. Open in any browser, Xcode, or a Markdown/SVG viewer.

| Diagram | Flow |
|---|---|
| [`room-generation-flow.svg`](room-generation-flow.svg) | Default Photo to 3D Room flow — `SinglePhotoRoomView` collects a camera/gallery image and sidecar metadata, `DepthAnythingRoomReconstructor` runs GeoCalib + Depth Anything + RTMDet object-anchor work, room measurement produces W×H×D, then the app exports a textured USDZ and opens `ModelViewerView`. |
| [`rtmdet-swift-flow.svg`](rtmdet-swift-flow.svg) | RTMDet instance segmentation ("brain") live/still loop — `RTMDetImageInference` + `FurnitureFitView` + Settings image scan. Camera/still image → Core ML image input → raw heads → confidence-first NMS → mask affinity → pixel-union cutout → overlay gestures/display. |

Legend: green = ANE/GPU (accelerated), blue = CPU, gray = camera/data, purple = file/web, orange =
display. ★ = accelerated stage, ☆ = scalar CPU stage.
