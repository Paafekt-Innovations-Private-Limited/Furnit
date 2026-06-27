# Furnit — iOS diagrams

Real SVG flow diagrams for the iOS (Swift) app. Open in any browser, Xcode, or a Markdown/SVG viewer.

| Diagram | Flow |
|---|---|
| [`rtmdet-swift-flow.svg`](rtmdet-swift-flow.svg) | RTMDet instance segmentation ("brain") live/still loop — `RTMDetImageInference` + `FurnitureFitView` + Settings image scan. Camera/still image → Core ML image input → raw heads → confidence-first NMS → mask affinity → pixel-union cutout → overlay gestures/display. |
| [`sharp-swift-flow.svg`](sharp-swift-flow.svg) | SHARP room reconstruction — `SHARPService` → `SharpRoomView` / `GaussianSplatView`. Single photo → Core ML Gaussian-splat → PLY + `.splatcache` → render → dimensions. |

Legend: green = ANE/GPU (accelerated), blue = CPU, gray = camera/data, purple = file/web, orange =
display. ★ = accelerated stage, ☆ = scalar CPU stage.
