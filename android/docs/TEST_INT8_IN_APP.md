# Testing ExecuTorch INT8 in the app

Do this **after** you have built and installed the app and (if not using packaged assets) pushed models to the device.

For the current output-validated Vulkan room-creation path, see
[`EXECUTORCH_VULKAN_KNOWN_GOOD_FLOW.md`](EXECUTORCH_VULKAN_KNOWN_GOOD_FLOW.md).

The guidance below is broader and includes older single-Part4b and CPU-oriented test paths.
For the current working Vulkan room-creation setup, treat the known-good flow document above as the source of truth.

**Which ExecuTorch native AAR is in this APK?**

- **`BuildConfig.EXECUTORCH_USE_VULKAN_AAR`** (Kotlin) and manifest meta-data **`com.furnit.executorch.USE_VULKAN_AAR`** match the **Gradle product flavor** (`etVulkan` vs `etCpu`), not a Settings toggle.
- **Vulkan / GPU path:** build and install the **etVulkan** variant, e.g.  
  `./gradlew :app:assembleEtVulkanDebug`  
  APK under `app/build/outputs/apk/etVulkan/debug/` → **`EXECUTORCH_USE_VULKAN_AAR=true`**.
- **CPU / XNNPACK (no Vulkan delegate in the AAR):**  
  `./gradlew :app:assembleEtCpuDebug`  
  APK under `app/build/outputs/apk/etCpu/debug/` → **`EXECUTORCH_USE_VULKAN_AAR=false`**. Use this when you rely on CPU-only portable graphs; Vulkan-delegated `.pte` on an etCpu APK often fails at runtime.
- **`./gradlew assembleDebug`** builds **both** flavors (two APKs). For a single APK, call **`assembleEtVulkanDebug`** or **`assembleEtCpuDebug`** explicitly.

Logcat on init: `BUILD_CONFIG EXECUTORCH_USE_VULKAN_AAR=…` under tag `ExecutorchInt8Sharp`.

**Where to push `.pte` on the device (separate folders):**

| APK flavor | External app folder (adb push) |
|------------|--------------------------------|
| **etCpu** | `/sdcard/Android/data/com.furnit.android/files/models_cpu/` |
| **etVulkan** | `/sdcard/Android/data/com.furnit.android/files/models_cpuvulkan_hybrid/` |

The app also mirrors into internal storage under the same subdir name (`files/models_cpu` or `files/models_cpuvulkan_hybrid`). Logcat prints both paths: `ExecuTorch model roots: internal=… external=…`.  
**Legacy `files/models/` is not used** — push only to `models_cpu` / `models_cpuvulkan_hybrid`.  
**Legacy `files/models_vulkan/`:** on launch the app copies `sharp_split*.pte` into `models_cpuvulkan_hybrid`, removes matching sources, and deletes the old folder when empty.

Helper scripts: `push_sharp_executorch_cpu_models.sh` → `models_cpu`; `push_sharp_cpuvulkan_hybrid_androidstudio.sh` / `push_sharp_vulkan_aar_compat.sh` → `models_cpuvulkan_hybrid`.  
To migrate an old `files/models/` tree on device: `android/migrate_legacy_models_to_cpu_vulkan.sh`.

---

## Yes — we're trying INT8 for the full pipeline

- **Part1, Part2, Part3, Part4a** in the C++ full pipeline are INT8 (those .pte files are `*_int8.pte`).
- **Part4b** can be **INT8 or FP32**: the C++ code loads **INT8** when `sharp_split_part4b_int8.pte` exists on device, otherwise it uses **FP32** (`sharp_split_part4b.pte`). So "single Part4b" = one decoder run; that run is INT8 if you have the INT8 Part4b file, else FP32.
- Right now almost everyone runs Part4b as **FP32** because the repo doesn’t export `sharp_split_part4b_int8.pte`. To actually run Part4b in INT8 you need to add that file to the models dir (export it from your pipeline and push, or drop it into `executorch_models/` and rebuild/push). Logcat will show `Part4b single: INT8` or `Part4b single: FP32` so you can confirm.

## Settings screen — general INT8 test options

| Setting | Value | Notes |
|--------|--------|--------|
| **Inference backend** | **ExecuTorch INT8** | Radio under Developer (required). |
| **CPU ExecuTorch INT8** vs Vulkan | Either | CPU path uses portable Part4b / tiles as on disk. |
| **Stable mode (prefer single Part4b)** | **ON (default)** | If `sharp_split_part4b_int8.pte` or `sharp_split_part4b.pte` exists, **skips 16-tile path** even when `tile_00` is present — reduces INT8 “foggy square” tiles. |
| **Swap tile NDC X/Y** | **OFF** unless misaligned | Passed into C++ tiled path only; fixes transposed tile layout. |
| **Debug mode** | Optional | For logcat (`PART4B_ROUTING`, `runPart4bTiledFullPipeline`). |

**Vulkan hybrid requirement:** With **ExecuTorch INT8 (Part1+2 + Vulkan — Hybrid)** selected (`executorch_int8_use_cpu_stable=false`), the app **refuses to initialize or infer** unless **portable Part1+2** sidecars exist (e.g. `sharp_split_part1_int8.pte` + `sharp_split_part2_int8.pte`, or matching `*_fp16` / `*_fp32` pairs). There is **no** Vulkan-only Part1+2 fallback.

**Routing:** `adb logcat -s ExecutorchInt8Sharp:I | grep PART4B_ROUTING` — fields `prefer_single=`, `path=…`. **Vulkan / hybrid (GPU Part4):** `prefer_single=false` always; native runs **tiled-first** (batched → sequential → single `part4b_vulkan` fallback). **CPU ExecuTorch stable (`executorch_int8_use_cpu_stable`):** `prefer_single=true` — **single .pte present → single first**; **only tiles on disk → tiled** (no choice).

Keep **Stable ON** and deploy a **single** Part4b `.pte` to avoid 16-tile INT8 fog when APK also ships `tile_00`.

**etCpu + portable Part4a (chunk 512 / 65) and Part4b:** heavier than Part3; you may see **tens of seconds** without new DEBUG lines while `forward()` runs. A full room on a fast phone is often **~2–3 minutes** total; slower devices take longer. Use **`sharp_executorch_full:I`** for milestones: `Part4a/512: mmap+load OK`, `forward finished`, then 65 and Part4b.

---

## 1. Set the backend

- Open **Settings** (gear icon).
- Under **Developer**, set **Inference backend** to **ExecuTorch INT8**.

## 2. Enable the C++ full pipeline (optional but recommended)

- In **Developer**, turn **C++ ExecuTorch INT8** **ON** so the full pipeline (Part1–4b) runs in native code.

## 3. Part4b mode (choose one)

- **Single Part4b (recommended for quality):** **Stable mode ON** (default) and ensure **`sharp_split_part4b_int8.pte`** or **`sharp_split_part4b.pte`** is on device. Log: `Part4b single: INT8` or `Part4b single: portable`.
- **Tiled Part4b (lower RAM, may show INT8 fog/seams):** Turn **Stable mode OFF**. C++ uses 16 tiles when `sharp_split_part4b_tile_00.pte` or `tile_full` exists. Compare **FP32 tile** vs **INT8 tile** exports if you see foggy 384×384 blocks.

### If you see “foggy squares” (priority order)

1. **Use single Part4b** (Stable ON + single `.pte` on device) — code default.
2. **Compare** same scene with **FP32** tile vs INT8 tile (export).
3. **Swap tile NDC X/Y** if patches are misaligned (not usually “fog”).
4. **Re-calibrate** INT8 Part4b for **tile crops** (export/quant issue).

## 4. Run a room from a photo

- From the main screen, start **room from single photo** (gallery or camera).
- Pick or take a photo and start generation.
- Wait for the run to finish; the 3D room view should appear.

## 5. Confirm in logs (on your machine)

**Use tag `sharp_executorch_full`** for C++ pipeline logs (patches, Part3, Part4a, Part4b tiled/single, TOTAL). Without it you only see Kotlin and miss the flow.

```bash
adb logcat -s sharp_executorch_full:D ExecutorchInt8Sharp:D SharpService:D -v time
```

- You should see **no** `0x12` / InvalidArgument errors.
- **Expected C++ flow (tiled Part4b):**  
  `1x patches (25): ...ms` → `0.5x patches (9): ...ms` → `0.25x patch: ...ms. All 35 patches: ...ms` → `Part3: ...ms` → `Part4a/512: ...` → `Part4a/65: ...` → `Released Part1+Part2 cache...` → `runPart4bTiledFullPipeline: using .../sharp_split_part4b_tile_00.pte` → `Part4b tile 1/16` … `Part4b tile 16/16` → `[TIMING] Part4b tiled (full pipeline C++): ...ms, 1179648 Gaussians` → `TOTAL pipeline (tiled Part4b): ...ms. Gaussians=1179648`.
- **Single Part4b:** look for `Part4b single: INT8` or `Part4b single: FP32` and `TOTAL pipeline: ...ms. Gaussians=...`.
- PLY write should complete; any crash or “output invalid” means something failed.

---

**If you have multiple devices:** use one target, e.g.  
`adb -s <device_id> install -r app/build/outputs/apk/etVulkan/debug/app-etVulkan-arm64-v8a-debug.apk`  
(or `.../etCpu/debug/app-etCpu-arm64-v8a-debug.apk` for CPU / XNNPACK)  
Then push models to **`models_cpuvulkan_hybrid`** or **`models_cpu`** as in the table above (not the old single `models/` folder, unless you rely on legacy fallback).  
and  
`adb -s <device_id> push ...` or `adb -s <device_id> logcat ...`.
