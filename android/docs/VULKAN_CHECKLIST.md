# Vulkan verification checklist

Use this before or when debugging ExecuTorch Vulkan (BackendFailed / error 32).

## 0. Verify requirements (ExecuTorch + Vulkan)

- [ ] **ExecuTorch version** — This app uses `org.pytorch:executorch-android-vulkan:1.1.0` (see `android/app/build.gradle`). ExecuTorch 1.1 is the current stable; 1.1.0 is appropriate. Run `./verify_executorch_vulkan_requirements.sh` from `android/` to print the version and checklist.
- [ ] **Vulkan 1.1** — ExecuTorch Vulkan backend requires **Vulkan 1.1** on the device ([ExecuTorch Android Vulkan](https://docs.pytorch.org/executorch/1.0/android-vulkan.html)).
- [ ] **Vulkan extensions (recommended for FP16/INT8)** — For FP16 and quantized models, check on device (e.g. Vulkan Hardware Capability Viewer) that these are available: **VK_KHR_16bit_storage**, **VK_KHR_8bit_storage**, **VK_KHR_shader_float16_int8**. Some devices report Vulkan 1.4 but still hit BackendFailed (32) due to driver/op issues (e.g. Mali-G715).

## 1. Device requirements

- [ ] **Vulkan 1.1+** — ExecuTorch Vulkan backend expects Vulkan 1.1 or above ([ExecuTorch Android Vulkan](https://docs.pytorch.org/executorch/1.0/android-vulkan.html)).
- [ ] **Check on device** — Use an app such as “Vulkan Hardware Capability Viewer” to confirm API version (e.g. 1.1.x or 1.4.x) and driver.
- [ ] **No Ultralytics in app** — This app does not call `check_executorch_requirements()`; use the [Ultralytics checks reference](https://docs.ultralytics.com/reference/utils/checks/) only in your own tooling if needed.

## 2. App build

- [ ] **Vulkan AAR** — `android/app/build.gradle` includes `implementation 'org.pytorch:executorch-android-vulkan:1.1.0'` (or your version).
- [ ] **Rebuild** — App was rebuilt after adding or changing the Vulkan dependency.

## 3. Model export

- [ ] **Exported with Vulkan** — Part1/Part2 (and Part3/Part4 if used) were exported with `--backend vulkan` so the `.pte` contains Vulkan delegates.
- [ ] **Verify delegate** — Run `python inspect_pte_delegates.py <path/to/sharp_split_part1_vulkan_fp16.pte>` and confirm output shows “VERIFIED: Vulkan delegate is present”.

## 4. Input size

- [ ] **Part1 input** — Part1 is fed **384×384 patches** only (cropped from the full image). Full image is 1536×1536 or 1280×1280 NCHW; pipeline validates length and Part1 shape before forward.
- [ ] **Reduce memory (optional)** — Use **1280×1280**: enable "Use 1280×1280" in Settings and export Part3/Part4 with `--image-size 1280` (e.g. `python export_sharp_executorch_split4.py --backend vulkan --chunked-part4 --dtype fp16 --image-size 1280`). Push the 1280 Part3/Part4 .pte files when using 1280.

## 5. Runtime

- [ ] **Models on device** — Vulkan `.pte` files are present in the app models dir (e.g. pushed via `adb push ... /sdcard/Android/data/com.furnit.android/files/models/`).
- [ ] **Portable fallback** — If Vulkan Part1 fails (e.g. BackendFailed 32 on Mali-G715), the app falls back to **Kotlin pipeline with portable (CPU) Part1/Part2** when `sharp_split_part1.pte` and `sharp_split_part2.pte` (or `_fp16.pte`) are present. Push portable Part1/2 for devices where Vulkan fails.
- [ ] **Logs** — On failure, check logcat for `Part1 fail`, `forward error 32`, `BackendFailed`; `BEFORE_PART1_FORWARD` / `AFTER_PART1_FORWARD` confirm input shape and forward status.

## 6. References

- [Ultralytics ExecuTorch guide](https://docs.ultralytics.com/integrations/executorch/)
- [Model Size Optimization](https://docs.ultralytics.com/integrations/executorch/#model-size-optimization)
- [ExecuTorch Vulkan backend](https://docs.pytorch.org/executorch/1.0/android-vulkan.html)
- `EXECUTORCH_VULKAN_SETUP.md` and `EXECUTORCH_VULKAN_OPS.md` in this repo.
