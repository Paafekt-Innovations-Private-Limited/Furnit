# Furnit — iOS diagrams

Real SVG flow diagrams for the iOS (Swift) app. Open in any browser, Xcode, or a Markdown/SVG viewer.

| Diagram | Flow |
|---|---|
| [`rtmdet-swift-flow.svg`](rtmdet-swift-flow.svg) | RTMDet instance segmentation ("brain") live loop — `RTMDetImageInference` + `FurnitureFitView`. Camera → Core ML (ANE) → decode/NMS → mask head → cutout → display. |
| [`sharp-swift-flow.svg`](sharp-swift-flow.svg) | SHARP room reconstruction — `SHARPService` → `SharpRoomView`. Single photo → Core ML Gaussian-splat → PLY → render → dimensions. |

Legend: green = ANE/GPU (accelerated), blue = CPU, gray = camera/data, purple = file/web, orange =
display. ★ = accelerated stage, ☆ = scalar CPU stage.
