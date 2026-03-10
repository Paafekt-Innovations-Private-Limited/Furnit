# Testing ExecuTorch INT8 in the app

Do this **after** you have built and installed the app and (if not using packaged assets) pushed models to the device.

---

## Yes — we're trying INT8 for the full pipeline

- **Part1, Part2, Part3, Part4a** in the C++ full pipeline are INT8 (those .pte files are `*_int8.pte`).
- **Part4b** can be **INT8 or FP32**: the C++ code loads **INT8** when `sharp_split_part4b_int8.pte` exists on device, otherwise it uses **FP32** (`sharp_split_part4b.pte`). So "single Part4b" = one decoder run; that run is INT8 if you have the INT8 Part4b file, else FP32.
- Right now almost everyone runs Part4b as **FP32** because the repo doesn’t export `sharp_split_part4b_int8.pte`. To actually run Part4b in INT8 you need to add that file to the models dir (export it from your pipeline and push, or drop it into `executorch_models/` and rebuild/push). Logcat will show `Part4b single: INT8` or `Part4b single: FP32` so you can confirm.

## Settings screen — what to set (INT8 C++ full pipeline)

| Setting | Value | Notes |
|--------|--------|--------|
| **Inference backend** | **ExecuTorch INT8** | Radio under Developer (required). |
| **C++ ExecuTorch INT8** | **ON** | Full pipeline Part1–4b in C++; Part4b is INT8 when the file exists, else FP32. |
| **Stable mode (single Part4b only)** | **ON** | One decoder run (not 16 tiles); that run is INT8 or FP32 as above. |
| **Part4b tiled (experimental)** | **OFF** | Use single Part4b path. |
| **Swap tile NDC X/Y** | **OFF** | Only if tiled layout is wrong. |
| **Debug mode** | Optional (ON for logs) | For logcat timing/debug. |

So: **ExecuTorch INT8** backend, **C++ ExecuTorch INT8 = ON**, **Stable mode = ON** to run the full INT8 pipeline with single Part4b (INT8 when `sharp_split_part4b_int8.pte` is present, FP32 otherwise).

---

## 1. Set the backend

- Open **Settings** (gear icon).
- Under **Developer**, set **Inference backend** to **ExecuTorch INT8**.

## 2. Enable the C++ full pipeline (optional but recommended)

- In **Developer**, turn **C++ ExecuTorch INT8** **ON** so the full pipeline (Part1–4b) runs in native code.

## 3. Part4b mode (choose one)

- **Single Part4b (stable):** Turn **Stable mode (single Part4b)** **ON**.  
  Uses one decoder run; if `sharp_split_part4b_int8.pte` is on device, log will show `Part4b single: INT8`, otherwise `Part4b single: FP32`.
- **Tiled Part4b (faster, experimental):** Turn **Part4b tiled (experimental)** **ON** and **Stable mode** **OFF**.  
  Uses 16-tile Part4b when tile models are present.

## 4. Run a room from a photo

- From the main screen, start **room from single photo** (gallery or camera).
- Pick or take a photo and start generation.
- Wait for the run to finish; the 3D room view should appear.

## 5. Confirm in logs (on your machine)

```bash
adb logcat -s sharp_executorch_full:D ExecutorchInt8Sharp:D SharpService:D -v time
```

- You should see **no** `0x12` / InvalidArgument errors.
- For C++ full pipeline single Part4b: look for `Part4b single: INT8` or `Part4b single: FP32` and a line like `Part4b (single): ...ms. TOTAL pipeline: ...ms. Gaussians=...`.
- PLY write should complete; any crash or “output invalid” means something failed.

---

**If you have multiple devices:** use one target, e.g.  
`adb -s <device_id> install -r app/build/outputs/apk/debug/app-debug.apk`  
and  
`adb -s <device_id> push ...` or `adb -s <device_id> logcat ...`.
