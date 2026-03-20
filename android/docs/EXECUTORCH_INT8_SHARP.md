## ExecuTorch INT8 SHARP backend

This backend runs the full SHARP room-generation pipeline using **ExecuTorch INT8** models on device. It is designed as a memory‑safe, mobile‑optimized implementation that avoids the 4 GB activation peaks of a monolithic decoder while keeping end‑to‑end latency low.

### Models and files

The ExecuTorch INT8 backend expects split `.pte` files under the app’s **`models_cpu`** directory (etCpu flavor): internal and external `files/models_cpu/`. See **`docs/EXECUTORCH_CPU_MODELS_SYNC.md`** for clear + push scripts and Part4b mismatch (error 18).

Legacy layouts may still resolve under `files/models/`; prefer **`models_cpu`** for the native full pipeline.

- **Encoder / feature stages (INT8):**
  - `sharp_split_part1_int8.pte`
  - `sharp_split_part2_int8.pte`
  - `sharp_split_part3_int8.pte`
- **Decoder / ViT chunks (FP32) + final decoder (FP32, optional INT8):**
  - `sharp_split_part4a_chunk_512.pte`
  - `sharp_split_part4a_chunk_65.pte`
  - `sharp_split_part4b.pte` (FP32, required fallback)
  - `sharp_split_part4b_int8.pte` (INT8, **optional**; when present, the C++ full pipeline prefers this over the FP32 file for Part 4b)

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

- **Part4b single vs 16-tile (CPU / C++):** By default **Settings → Stable mode (prefer single Part4b)** is **ON**, so if `sharp_split_part4b_int8.pte` or `sharp_split_part4b.pte` is on device, the pipeline uses **one** Part4b forward even when tiled `.pte` files are packaged — this avoids common **INT8 + 16-tile** “foggy square” artifacts. Turn Stable **OFF** to force the tiled path when tile models exist (lower RAM). See `docs/TEST_INT8_IN_APP.md`.

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

