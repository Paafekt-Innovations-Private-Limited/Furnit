# Furnit Android

Android app for Furnit (3D room models, SHARP inference, etc.).

## SHARP inference: backend comparison

Typical end-to-end time for one room generation (single photo → 3D Gaussian room) on device:

| Backend | Total | Part 1+2 (encoders) | Part 3 (image) | Part 4 (decoder) | PLY write |
|--------|-------|----------------------|----------------|------------------|-----------|
| **ExecuTorch INT8** | **~2 min 15 s** | ~43 s | ~1 s | ~86 s | ~4 s |
| **ExecuTorch FP16** | **~3 min 20 s** | ~93 s | ~2.5 s | ~97 s | ~5 s |
| **ExecuTorch FP32** | **~4 min 15 s** | ~2 min 27 s | ~4 s | ~96 s | ~7 s |
| **Split ONNX FP32** | **~5 min 17 s** | P1 ~101 s, P2 ~34 s | ~5 s | ~176 s | ~1 s |
| **Native .pt (scripted)** | **~5 min 41 s** | P1 ~82 s, P2 ~100 s | ~7.5 s | ~135 s | ~4 s |
| **LiteRT (TFLite FP16)** | **~8–9 min (est.)** | P1 ~170 s, P2 ~226 s (35 pat. each) | ~6 s | P4a ~6 s, P4b est. ~1.5–2 min | — |
| **NCNN (component)** | **~24 min** | 35 pat. ~77 s/pat (~22 min) | ImageEnc ~46 s | in pipeline | ~1 s |
| **Split ONNX FP16** | **N/A** | — | — | — | — |
| **ONNX INT8 (single)** | **~10 min 23 s** | single model (no part breakdown) | — | — | ~3 s |

- **Part 1+2 (ExecuTorch)**: Encode 35 patches (1x + 0.5x + 0.25x). INT8 ~2.2× faster than FP16 (43 s vs 93 s), FP16 ~1.6× faster than FP32 (93 s vs 147 s).
- **Part 1+2 (Split ONNX)**: Chunked Part 1 (1a + 1b) does full patch encoder; Part 2 does merge/rest. Saves intermediates to disk between parts.
- **NCNN (component)**: Vulkan component mode, 35 patches (~77 s each) then ImageEncoder + GaussianHead. Outputs **147456 Gaussians** (not 1.18M); **quality is wrong** (e.g. blue blob). Use other backends for correct room mesh.
- **Native .pt (scripted)**: PyTorch Mobile 4-part split (sharp_scripted_part1–4.ptl), load-on-demand. Output can be shakier/noisier than ExecuTorch or ONNX backends.
- **LiteRT (TFLite FP16)**: 4-part split (sharp_part1–4_fp16.tflite), single-patch mode: Part 1 and Part 2 each run 35 patches on CPU (XNNPACK), so P1+P2 dominate (~6 min 36 s). Merge ~0.7 s. Log ended before Part 4b/PLY; total estimated ~8–9 min.
- **Split ONNX FP16**: Not runnable on current ORT: `CPUExecutionProvider` has no kernel for `com.microsoft.Gelu` with `tensor(float16)` (only `tensor(float)`). No in-app fix without ORT adding FP16 Gelu or using another EP.
- **ONNX INT8 (single)**: One ~715 MB INT8 ONNX graph; inference ~620 s, no per-part timing. Slower than split/ExecuTorch due to single-session memory and CPU execution.
- **Part 3**: One full-image encoder pass.
- **Part 4**: Chunked 4a + 4b. 4b is FP32 decoder and dominates (Conv/FusedConv ~96% of Part4b time per ORT profile).
- **PLY write**: ~293 MB room file to disk.

**Summary:** ExecuTorch INT8 is fastest (~2 min 15 s); ExecuTorch FP16 ~3 min 20 s; ExecuTorch FP32 ~4 min 15 s; Split ONNX FP32 ~5 min 17 s; Native .pt ~5 min 41 s (scripted, can be shaky); LiteRT (TFLite FP16) ~8–9 min est. (P1+P2 per-patch CPU); NCNN ~24 min (component mode; output quality wrong—blue blob); Split ONNX FP16 N/A (ORT CPU lacks FP16 Gelu); ONNX INT8 single ~10 min 23 s (one large graph, no part splitting). Part 4 (decoder) is the main bottleneck on split backends.

Logs: `adb logcat -s ExecutorchInt8Sharp:D SharpService:D -v time` (ExecuTorch INT8), `ExecutorchFp16Sharp:D SharpService:D` (ExecuTorch FP16), `ExecutorchSharp:D SharpService:D` (ExecuTorch FP32), `SplitOnnxSharp:D SharpService:D` (Split ONNX FP32), `NativePtSharp:D SharpService:D` (Native .pt), `LiteRTSharp:D SharpService:D` (LiteRT), `SharpNCNN:V NcnnSharp:D SharpService:D -v time` (NCNN), `SplitOnnxFp16Sharp:D SharpService:D` (Split ONNX FP16), `OnnxInt8Sharp:D SharpService:D` (ONNX INT8 single), `TorchMobileSharp:D SharpService:D` (PyTorch Mobile).

### Settings: PyTorch Mobile vs Native .pt

In **Settings → Developer → Inference backend**, the last two options are:

- **PyTorch Mobile** (second-to-last): Uses **TorchMobileSharp**. Single full-graph model **`sharp_mobile.ptl`** (~2.5 GB), loaded with PyTorch Android **LiteModuleLoader** (`org.pytorch`). Same weights as Python; no ONNX/ExecuTorch/TFLite conversion. Input `[1, 3, 1536, 1536]` float32 → output `[N, 14]` Gaussian parameters. If `sharp_mobile.ptl` is not found, the app falls back to ONNX (e.g. Split ONNX if available); logcat shows: `PyTorch Mobile model not found. Falling back to ONNX.`
- **Native .pt** (last): Uses **NativePtSharp**. **4-part split** (`sharp_scripted_part1.ptl` … `sharp_scripted_part4.ptl`), load-on-demand via LibTorch native runtime. TorchScript + C++ path; output can be shakier than ExecuTorch/ONNX.

## Removed components

### CameraClassifierActivity (SmartyPants)

**Removed:** The live camera classifier feature and its entry point have been removed from the app.

**What it did:**

- **CameraClassifierActivity** was a full-screen activity that showed a live camera preview with an on-device object-detection overlay (SmartyPants). It used **CameraX** for preview and **ExecuTorch** with a **MobileNetV3** classification model to run inference on each frame (~10 FPS), showing top-5 class labels and an FPS counter.
- It was launched from a camera icon (📷) in the home screen top bar (ContentActivity). That icon has been removed.
- Supporting code that was removed together with it:
  - **ExecutorchClassifier** – loaded and ran the ExecuTorch MobileNetV3 `.pte` model and returned classification results.
  - **FrameAnalyzer** – CameraX `ImageAnalysis.Analyzer` that throttled frames and called the classifier, then passed results to the UI.
  - **ImagePreprocessor** – converted CameraX `ImageProxy` (YUV) to bitmap and preprocessed for MobileNetV3 (224×224, ImageNet normalization, NCHW `FloatArray`).

Room creation from a **single photo** (take or pick image → AI or manual 3D room) is unchanged and is still available via the gallery/image icon (🖼) on the home screen; that flow uses `SinglePhotoRoomActivity`, not the removed camera classifier.
