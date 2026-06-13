# Furnit — Android diagrams

Real SVG flow diagrams for the Android (Kotlin) app. Open in any browser or Android Studio.

| Diagram | Flow |
|---|---|
| [`rtmdet-android-flow.svg`](rtmdet-android-flow.svg) | RTMDet instance segmentation ("brain") live loop — `FurnitureFitManager` + `SharpRoomActivity`. CameraX → ONNX Runtime → decode/NMS → mask build → cutout → display. |
| [`sharp-android-flow.svg`](sharp-android-flow.svg) | SHARP room reconstruction — `SharpService` → `ExecutorchInt8Sharp`. Single photo → sliding-pyramid ExecuTorch (.pte) → Gaussian-splat → PLY → render → dimensions. |

Legend: green = ExecuTorch/ONNX accelerator (Vulkan/XNNPACK), blue = Kotlin CPU, gray = camera/data,
purple = file/web, orange = display. ★ = accelerated stage, ☆ = scalar CPU stage.

Written Android docs live in [`android/docs/`](../docs/).
