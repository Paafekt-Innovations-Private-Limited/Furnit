## ExecuTorch INT8 SHARP backend

This backend runs the full SHARP room-generation pipeline using **ExecuTorch INT8** models on device. It is designed as a memory‑safe, mobile‑optimized implementation that avoids the 4 GB activation peaks of a monolithic decoder while keeping end‑to‑end latency low.

### March 21, 2026 settings cleanup

For the Settings cleanup that removed some ExecuTorch-facing toggles from the Android UI, **no changes were made inside ExecuTorch itself**.

- No changes to the ExecuTorch runtime or AARs.
- No changes to vendored/native ExecuTorch source.
- No changes to Vulkan delegate registration or kernels.
- No changes to exported `.pte` model files.

The change was app-layer only:

- `SettingsActivity.kt` no longer shows `Use true 1280x1280`, `Prefer Vulkan FP16 models`, or `Prefer single Part4b`.
- `ExecutorchInt8Sharp.kt` now uses fixed app values for those settings:
  - `Use true 1280x1280 = OFF`
  - `Prefer Vulkan FP16 models = ON`
  - `Prefer single Part4b = OFF`
- `ExecutorchFixedSettings.kt` syncs those fixed values into `furnit_prefs` so existing installs stay aligned with the hidden defaults.

### Current model files by flow (March 21, 2026)

These tables describe the **current app/runtime flow**, not just every filename the code can recognize.

#### Vulkan flow (`etVulkan`, `files/models_vulkan`)

| Stage | Model file(s) used in the current flow | Notes |
|---|---|---|
| Part1 | `sharp_split_part1_vulkan_fp32.pte` | Current `etVulkan` runtime resolves FP32 first. `_vulkan_fp16.pte` exists only as a fallback because the bundled Vulkan AAR does not include all FP16 conversion shaders. |
| Part2 | `sharp_split_part2_vulkan_fp32.pte` | Same FP32-first resolution as Part1. |
| Part3 | `sharp_split_part3_vulkan_fp32.pte` | `_vulkan_fp16.pte` is fallback only. `1280` Part3/Part4 files are not used because true 1280 is fixed OFF. |
| Part4a / 512-token chunk | `sharp_split_part4a_chunk_512_vulkan.pte` | Required. |
| Part4a / 65-token chunk | `sharp_split_part4a_chunk_65_vulkan.pte` | Required. |
| Part4b / tile_00 `stage_pre` | `sharp_split_part4b_tile_00_stage_pre_vulkan.pte` | Current known-good Vulkan path uses the fine-split `tile_00` route. |
| Part4b / tile_00 `decoder_head` | `sharp_split_part4b_tile_00_decoder_head.pte` | Vulkan stage in the current known-good path. |
| Part4b / tile_00 `init_base` | `sharp_split_part4b_tile_00_init_base.pte` | Portable stage in the current known-good path. |
| Part4b / tile_00 `raw_heads` | `sharp_split_part4b_tile_00_raw_heads_vulkan.pte` | Vulkan stage in the current known-good path. |
| Part4b / tile_00 `compose` | `sharp_split_part4b_tile_00_compose.pte` | Portable stage in the current known-good path. |

`sharp_split_part4b_vulkan.pte` still exists as a fallback single-decoder file, but it is **not** the current known-good Vulkan route.

#### CPU flow (`etCpu` or CPU ExecuTorch INT8, `files/models_cpu`)

| Stage | Model file(s) used in the current flow | Notes |
|---|---|---|
| Part1 | `sharp_split_part1_int8.pte` | If missing, native falls back to `sharp_split_part1.pte`, then `sharp_split_part1_fp16.pte`. |
| Part2 | `sharp_split_part2_int8.pte` | If missing, native falls back to `sharp_split_part2.pte`, then `sharp_split_part2_fp16.pte`. |
| Part1+2 optional batch helper | `sharp_split_part1_b4_int8.pte` and `sharp_split_part2_b4_int8.pte` | Used only if both files are present and the run is not forced into single-patch Part1/2 mode. |
| Part3 | `sharp_split_part3_int8.pte` | If missing, native falls back to `sharp_split_part3_fp16.pte`, then `sharp_split_part3.pte`. |
| Part4a / 512-token chunk | `sharp_split_part4a_chunk_512.pte` | Required. |
| Part4a / 65-token chunk | `sharp_split_part4a_chunk_65.pte` | Required. |
| Part4b single | `sharp_split_part4b.pte` | CPU current flow prefers a single Part4b decoder when present. Fallback order is `sharp_split_part4b.pte`, then `sharp_split_part4b_fp16.pte`, then `sharp_split_part4b_int8.pte`. |
| Part4b tiled fallback | `sharp_split_part4b_tile_b2.pte`, `sharp_split_part4b_tile_b4.pte`, `sharp_split_part4b_tile_00.pte`, `sharp_split_part4b_tile_full.pte` | Used only if no single Part4b file is available, or if the single path fails and native falls back to tiled routing. |

### Models and files

The ExecuTorch INT8 backend expects split `.pte` files under the app’s **`models_cpu`** directory (etCpu flavor): internal and external `files/models_cpu/`. See **`docs/EXECUTORCH_CPU_MODELS_SYNC.md`** for clear + push scripts and Part4b mismatch (error 18).

Legacy layouts may still resolve under `files/models/`; prefer **`models_cpu`** for the native full pipeline.

- **Encoder / feature stages (INT8):**
  - `sharp_split_part1_int8.pte`
  - `sharp_split_part2_int8.pte`
  - `sharp_split_part3_int8.pte`
- **Decoder / ViT chunks (FP32) + final decoder (FP32 / FP16 / INT8 fallback):**
  - `sharp_split_part4a_chunk_512.pte`
  - `sharp_split_part4a_chunk_65.pte`
  - `sharp_split_part4b.pte` (FP32, current CPU-preferred single decoder)
  - `sharp_split_part4b_fp16.pte` (FP16, optional single-decoder fallback)
  - `sharp_split_part4b_int8.pte` (INT8, optional single-decoder fallback)

`ExecutorchInt8Sharp` searches **`filesDir/models_cpu`** then **`getExternalFilesDir("models_cpu")`**, then legacy **`models`** paths. External `sharp_split*.pte` are synced into internal `models_cpu` for fast mmap.

### High‑level pipeline

The core implementation lives in `ExecutorchInt8Sharp`:

- **Preprocessing**
  - Input bitmap is resized to `1536×1536` and converted to NCHW floats using direct `ByteBuffer`/`FloatBuffer` for minimal allocations.
  - A reusable patch buffer (`384×384`) and full‑image buffer (`1536×1536`) are used to avoid per‑patch allocations and reduce GC pressure.

- **Encoder (Parts 1 & 2) – 35‑patch multi‑scale**
  - **1× scale:** 5×5 grid (`GRID_1X = 5`) over the full image; merged feature maps are 96×96 (`M_1X`).
  - **0.5× scale:** 3×3 grid (`GRID_05X = 3`) over a half‑size image; merged `x1Feat` is 48×48 (`M_05X`).
  - For each patch:
    - **Part 1 (`sharp_split_part1_int8.pte`)** runs the local encoder and produces:
      - Token sequence (`[1, 577, 1024]`)
      - Spatial feature map (`[1, 1024, 24, 24]`)
    - Spatial features and tokens are reshaped (`reshapeToSpatial`) into `[C, H, W]` and merged into three large feature volumes:
      - `latent0`, `latent1`, and `x0Feat` of shape `[1024, mSize1x, mSize1x]`
    - **Part 2 (`sharp_split_part2_int8.pte`)** refines the tokens; the outputs are merged into `x0Feat` using `mergeCrop`, which implements the exact crop/concat merge from the original Python SHARP encoder.

- **Image encoder (Part 3)**
  - **Part 3 (`sharp_split_part3_int8.pte`)** takes the full‑resolution image tensor `[1, 3, 1536, 1536]` and produces image tokens `[1, 577, 1024]`.

- **Part4b routing (current app behavior):**
  - **Vulkan:** the current known-good path is the tiled fine-split `tile_00` route listed in the Vulkan table above. The single `sharp_split_part4b_vulkan.pte` file is fallback only.
  - **CPU:** when a single decoder file exists, the CPU path prefers that single Part4b file over tiled artifacts. If no single decoder file exists, it falls back to tiled `tile_b2` / `tile_b4` / `tile_00` / `tile_full`.

- **Chunked decoder (Part 4)**
  - To avoid a single ~4 GB decoder activation peak, the decoder is split:
    - `sharp_split_part4a_chunk_512.pte` processes the first `512` tokens.
    - `sharp_split_part4a_chunk_65.pte` processes the remaining `65` tokens.
  - Outputs from the two chunks are concatenated back to `[1, 577, 1024]`.
  - **Part 4b (`sharp_split_part4b.pte`)** takes:
    - The concatenated tokens `[1, 577, 1024]`
    - Full image tensor `[1, 3, 1536, 1536]`
    - The merged feature volumes `latent0`, `x0Feat` (1× scale, 96×96), and `x1Feat` (0.5× scale, 48×48)
  - It returns final Gaussian parameters `[N, 14]`:
    - `(x, y, z)`
    - opacity
    - scales `(sx, sy, sz)`
    - rotation quaternion `(rw, rx, ry, rz)`
    - DC color coefficients and remaining SH coefficients.

- **PLY streaming**
  - `writePly` converts `[N, 14]` into a binary **PLY** point cloud / Gaussian file:
    - Positions `(x, y, z)` with axis conventions applied.
    - SH color coefficients, with DC terms scaled by `SH_C0`.
    - Opacity mapped through a precomputed logit LUT (`LOGIT_LUT`).
    - Scales passed through `lnLut` to avoid computing `ln` per vertex.
    - Rotations normalized to unit quaternions.
  - A reusable `plyBatch` direct buffer and `zeroSHBuffer` avoid per‑vertex allocations.

### Performance characteristics

- **Quantization and kernels**
  - Parts 1–3 are **INT8 ExecuTorch models**, targeting ARM NEON INT8 kernels; this reduces memory bandwidth and improves throughput compared to FP32.
  - The decoder chunks and final decoder are FP32, but the heavy encoder and ViT blocks are quantized.

- **End‑to‑end latency**
  - On a modern high‑end Android device, a full single‑photo → 3D room run typically completes in **~1.5–3 seconds**, depending on:
    - Device CPU performance and thermal throttling.
    - Number of Gaussians (`N`) and scene complexity.
    - Background load (other apps, power saver mode, etc.).
  - This is significantly faster than the FP32 ExecuTorch and ONNX backends and is intended as the **fast path** for SHARP on mobile.

> Note: Actual timings will vary by device. Use `adb logcat -s ExecutorchInt8Sharp:D SharpService:D -v time` to capture real measurements on your target hardware.

### Initialization and usage

- The backend is exposed via `ExecutorchInt8Sharp`:
  - Call `ExecutorchInt8Sharp.getInstance(context)` to get the singleton.
  - Call `initialize()` once before inference; this simply marks the instance as ready.
  - Call `inferStreaming(bitmap, progressCallback)` to run the full pipeline and obtain a `StreamingResult`:
    - `plyFile` and `classicPlyFile` (currently identical) point to the generated room PLY.
    - `gaussianCount` is the number of Gaussians.
    - `roomWidth`, `roomHeight`, `roomDepth` are derived from the XYZ bounds.

- The integration in `SharpService`:
  - Treats ExecuTorch INT8 as a selectable backend (`executorch_int8`).
  - Does not perform a heavy preload step; models are loaded on demand inside `inferStreaming`.
  - Uses the same `StreamingResult` contract as other backends, so higher‑level UI code does not need to change.

### Testing (export, push, build, install, run)

1. **Export models** (from `android/`; ensure enough disk space):
   - INT8 parts 1–3:  
   `python export_sharp_executorch_int8_split4.py`  
   Output: `executorch_int8_models/sharp_split_part1_int8.pte`, `part2_int8.pte`, `part3_int8.pte`.
   - Chunked Part 4 (if not already present):  
   `python export_sharp_executorch_split4.py --chunked-part4`  
   Output in `executorch_models/`: `sharp_split_part4a_chunk_512.pte`, `sharp_split_part4a_chunk_65.pte`, `sharp_split_part4b.pte`.
   - **Optional: ExecuTorch INT8 Part 4b for C++ full pipeline:**  
   Export or place `sharp_split_part4b_int8.pte` into the same `executorch_models/` (or `executorch_int8_models/`) folder. On device, when this file is present next to `sharp_split_part4b.pte`, the C++ full INT8 pipeline will automatically run INT8 Part 4b; when it is absent, the FP32 `sharp_split_part4b.pte` is used as a safe fallback.

2. **Push models to device** (optional if using packaged APK)  
   `./push_sharp_executorch_int8_models.sh`  
   Pushes from `executorch_int8_models/` and `executorch_models/` to  
   `/sdcard/Android/data/com.furnit.android/files/models/`.

   **Or package in APK for testing:**  
   Before building, ensure the 6 .pte files exist in `executorch_int8_models/` and `executorch_models/`.  
   The Gradle task `copyExecutorchModelsIntoAssets` runs before each build and copies them into `app/src/main/assets/models/`.  
   The app copies from assets to internal storage on first run, so a friend can install the APK and use **Settings → ExecuTorch INT8** without pushing models manually. (APK size increases.)

3. **Build and install**  
   `./gradlew :app:assembleDebug`  
   `adb install -r app/build/outputs/apk/debug/app-debug.apk`

4. **Run test**  
   - In the app: **Settings → Developer → Inference backend → ExecuTorch INT8**.
   - Create a room from a single photo (gallery or camera).
   - Watch logs:  
     `adb logcat -s ExecutorchInt8Sharp:D SharpService:D -v time`  
   - Confirm no `0x12` / InvalidArgument errors and that PLY is written.
