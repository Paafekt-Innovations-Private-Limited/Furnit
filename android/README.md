# Furnit Android

Android app for Furnit (3D room models, SHARP inference, etc.).

## SHARP models: local (fast) vs friend APK (bundled)

**Default (`skipExecutorchAssets=true` in `gradle.properties`):** Android Studio Run is **fast**; the APK does **not** include `.pte`. Push models to the device (below) for SHARP.

**Friend APK with models inside:** from `android/` run:
```bash
./assemble_friend_apk_with_models.sh
# or: ./gradlew :app:assembleEtVulkanDebug -PskipExecutorchAssets=false
```
The script also **copies** the built APK(s) to **`friend-apk-dist/`** with a timestamp (`friend-YYYYMMDD-HHMMSS-*.apk`) so **Studio’s** output under `app/build/outputs/apk/` stays the one you use for Run/Debug.

See `docs/MODELS_APK_BUNDLE_TESTING.md` for hybrid/minimal flags and Zip32 limits.

## Smaller APK for sharing (arm64-only, no embedded SHARP models)

Typical **local debug** APK is already small because models are **not** bundled by default. To force a build **without** embedding even when you temporarily set `skipExecutorchAssets=false`:

1. **Build without embedding the SHARP models:**
   ```bash
   ./gradlew :app:assembleEtVulkanDebug -PskipExecutorchAssets=true
   ```
   Output under `app/build/outputs/apk/etVulkan/debug/`

2. **On the device**, SHARP (AI room from photo) will need the models pushed once. **etCpu** APK uses **`files/models_cpu/`**; **etVulkan** uses **`files/models_vulkan/`** (see `docs/TEST_INT8_IN_APP.md`). Stage copies under repo **`android/models_cpu/`** (see `models_cpu/README.md`) then push. Example for **CPU** / portable `.pte`:
   ```bash
   adb shell mkdir -p /sdcard/Android/data/com.furnit.android/files/models_cpu
   adb push executorch_int8_models/*.pte /sdcard/Android/data/com.furnit.android/files/models_cpu/
   adb push executorch_models/sharp_split_part4a_chunk_512.pte /sdcard/Android/data/com.furnit.android/files/models_cpu/
   adb push executorch_models/sharp_split_part4a_chunk_65.pte /sdcard/Android/data/com.furnit.android/files/models_cpu/
   adb push executorch_models/sharp_split_part4b.pte /sdcard/Android/data/com.furnit.android/files/models_cpu/
   # Optional: INT8 Part4b (C++ prefers when present)
   adb push executorch_models/sharp_split_part4b_int8.pte /sdcard/Android/data/com.furnit.android/files/models_cpu/
   ```
   Or copy the same files onto the device into that path (e.g. via file manager). Without these, the app runs but “Create room from photo” (AI) will report missing models.

3. **ABI split** is already enabled (arm64-v8a only), so the APK only contains one architecture.

## Build: Gradle daemon / wildcard IP on macOS

If the build fails with **"Could not determine a usable wildcard IP for this machine"** and/or **"xargs: sysconf(_SC_ARG_MAX) failed"**, try:

1. **Use Android Studio’s JDK** (often fixes the IP issue):
   ```bash
   export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
   ./gradlew :app:assembleDebug
   ```
   (On older Android Studio the JRE path may be `.../jre/jdk/Contents/Home`.)

2. **Disable the daemon** (avoids daemon socket/lock setup that triggers the error):
   - In `gradle.properties` add: `org.gradle.daemon=false`
   - Or run once: `./gradlew :app:assembleDebug --no-daemon`

3. **Ask Ultralytics AI** for more suggestions: install Playwright browsers (`playwright install`), then:
   ```bash
   python3 ultralytics_ask_ai.py --gradle --headless
   ```
   Response is printed and saved to `ultralytics_response.txt`; use it in Cursor and confirm before applying changes.

## Viewing logs

With the device connected (or emulator running), use `adb logcat`. If more than one device is connected, target one with `-s <device_serial>` (see `adb devices`).

**3D viewer (camera framing):** To see WebView camera/Box3 logs, include `SharpRoomActivity` in logcat (e.g. `adb logcat -s ExecutorchInt8Sharp:D SharpService:D SharpRoomActivity:D -v time`). Look for `[SharpRoom] CAMERA_FRAME` with isPortrait, boxMin/Max, center, roomW/H/D, inset, camPos, target. On open, Kotlin logs `SharpRoom intent roomWidth=... isPortrait=...`.

**ExecuTorch INT8 pipeline timing:** The C++ full pipeline logs under tag **`sharp_executorch_full`** (matches the .so name); include it or you won’t see patches/Part3/Part4a/Part4b/TOTAL. Example from a typical run (~3 min total):
- 1x patches (25): ~36 s  
- 0.5x patches (9): ~14 s  
- 0.25x patch: ~1.5 s  
- Part3 (image encoder): ~2 s  
- Part4a (2 chunks): ~4 s  
- **Part4b (decoder + Gaussian head): ~106 s** (single) or **Part4b tiled: ~86 s** (16 tiles) ← dominant  
- writePly (1.18M Gaussians): ~14 s  
```bash
adb logcat -s sharp_executorch_full:D ExecutorchInt8Sharp:D SharpService:D -v time
```

**Room list / 3D room view (camera position, model load):**
```bash
adb logcat -s ModelDetailActivity:D RoomBoundaryManager:D -v time
```

**Room picking (tap room name → correct room opens; debug vintage vs correct room):**
```bash
adb logcat -s ContentActivity:D ModelManager:D ModelDetailActivity:D FurnitureFit:D SharpRoomActivity:D GLBRoomActivity:D -v time
```

**Room position / camera (room only right edge visible; centering):**
```bash
adb logcat -s ModelDetailActivity:D RoomBoundaryManager:D -v time
```
Shows model bbox center/extents, model position, room bounds, and camera pos/lookAt when opening a room from the list.

**Camera debug: which path ran and final position (capture to file):**
Use this to see whether GLB (RoomBoundaryManager / ModelDetail / FurnitureFit) or PLY (SharpRoomActivity WebView) set the camera, and what values were used.
```bash
adb logcat -s RoomBoundaryManager:D ModelDetailActivity:D FurnitureFit:D SharpRoomActivity:D -v time 2>&1 | tee camera_debug.log
```
Then open the room (from list or create from photo). Look for:
- `[BackCenter]` = RoomBoundaryManager computed position (GLB path).
- `[ModelDetail] getCameraAtBackCenter CALLED` / `camera SET` = opening a GLB room from the list.
- `[FurnitureFit] getCameraAtBackCenter CALLED` / `camera SET` = GLB room as brain/segmentation background.
- `[SharpRoom] Building WebView HTML` = PLY/splat viewer started (photo-orientation and isPortrait).
- `CAMERA_POSITION_FINAL` (in WebGL message) = JS in SharpRoomActivity set the camera; shows pos, depthAlongView, insetFraction, insetFromBack.
If you only see `[SharpRoom]` and `CAMERA_POSITION_FINAL`, the camera is controlled by the WebView JS (SharpRoomActivity), not Kotlin RoomBoundaryManager.

**WebView GLB viewer (when opening a GLB room from list):**
```bash
adb logcat -s chromium:D -v time
```
(WebView logs may use the `chromium` tag; check logcat for `[GLBViewer]` in the message.)

**Camera and brain icon (room list / FurnitureFit):**  
Room list 3D view uses the same virtual camera as iOS: **back-left corner** inside the room (RealityKitBoundaryManager), looking at the front wall. When you tap the **brain icon**, the app starts FurnitureFit (furniture segmentation) and passes **ROOM_FOLDER** and **ROOM_ID** so the 3D background matches the opened room when possible.

**How the brain (segmentation) background is picked:**  
When you tap the brain from an opened room: (1) If the room folder has **room.glb**, it is shown as the 3D background (SceneView). (2) If the room folder has only **room.ply** (SHARP Gaussian splat), the PLY is shown as the background via a WebView (same SparkJS viewer as the Sharp room screen). (3) Bundled rooms (`vintage`, `cozy_room`) or rooms found by **ROOM_ID** under `rooms/` or `sharp_rooms/` use their `room.glb` when present. No fallback: if there is no `room.glb` and no `room.ply`, no 3D backdrop is shown.

**Logs for brain icon / background:**
```bash
adb logcat -s FurnitureFitActivity:D FurnitureFit:D -v time
```

**SHARP / ExecuTorch INT8:**
```bash
adb logcat -s ExecutorchInt8Sharp:D SharpService:D -v time
```

**Multiple tags:** e.g. room view + SHARP:
```bash
adb logcat ModelDetailActivity:D RoomBoundaryManager:D ExecutorchInt8Sharp:D SharpService:D -v time
```

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

For details on the ExecuTorch INT8 backend, model files, and **testing steps** (export, push, build, install, run), see [android/docs/EXECUTORCH_INT8_SHARP.md](android/docs/EXECUTORCH_INT8_SHARP.md).

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
