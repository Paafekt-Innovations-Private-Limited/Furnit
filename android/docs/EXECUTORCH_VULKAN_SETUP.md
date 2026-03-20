# Making Vulkan work

**Verify requirements (not yet implemented)** — The app does **not** call Ultralytics’ `check_executorch_requirements()`. ExecuTorch’s Vulkan backend expects **Vulkan 1.1+** on the device ([ExecuTorch Vulkan backend](https://docs.pytorch.org/executorch/1.0/android-vulkan.html)). To verify device support you can use the [Ultralytics checks reference](https://docs.ultralytics.com/reference/utils/checks/) in your own tooling, or run a Vulkan-capability app (e.g. “Vulkan Hardware Capability Viewer”) on the device. A runtime preflight that checks Vulkan availability before loading models has not been added.

**Reduce input size (not tried)** — We have **not** tried a smaller `imgsz` (e.g. 320) to reduce memory pressure. The pipeline is fixed at **1536×1536** input: `IMAGE_SIZE=1536`, `PATCH_SIZE=384`, grid and strides are hardcoded in C++ and Kotlin. Supporting a smaller size would require (1) re-exporting SHARP parts for the target resolution (or accepting quality loss by resizing 320→1536 before inference), (2) making `IMAGE_SIZE` and related constants configurable, and (3) updating grid/stride math. See [Model Size Optimization](https://docs.ultralytics.com/integrations/executorch/#model-size-optimization) for the general idea.

1. **App dependency** — The app must include the Vulkan delegate AAR so the runtime can execute Vulkan-partitioned `.pte` files:
   - `android/app/build.gradle`: `implementation 'org.pytorch:executorch-android-vulkan:1.1.0'`
   - Rebuild the app after adding it.

2. **Export with Vulkan** — Models must be exported with the Vulkan backend so the `.pte` contains Vulkan delegates.
   - **Vulkan FP16 + B2 (recommended, ~95% success, avoids INT8 staging crash):**  
     `WEIGHTS=/path/to/sharp.pt DTYPE=fp16 PATCH_BATCH=2 ./export_sharp_executorch_vulkan_full.sh`  
     Produces `sharp_split_part1_vulkan_fp16.pte`, `part2_vulkan_fp16.pte`, `part3_vulkan_fp16.pte`, `part1_b2_vulkan_fp16.pte`, `part2_b2_vulkan_fp16.pte`, plus chunked Part4. The C++ pipeline prefers these when Vulkan is enabled.
   - **INT8 Vulkan (faster when it works, ~20% success on some devices):**  
     `python export_sharp_executorch_int8_split4.py --backend vulkan --weights /path/to/sharp.pt --output-dir executorch_int8_models`
   - **Chunked Part4 (FP32 Vulkan):**  
     `./export_sharp_executorch_vulkan_full.sh`  
     Writes `executorch_models/sharp_split_part4a_chunk_512.pte`, `sharp_split_part4a_chunk_65.pte`, `sharp_split_part4b.pte`.

3. **Push to device** — Push the exported `.pte` files (e.g. `./push_sharp_executorch_int8_models.sh` and/or `./push_sharp_executorch_models.sh executorch_models`). Include Vulkan FP16 Part1/2/3 and B2 in the same models dir so the app can load them when Vulkan is on.

4. **Runtime** — When Vulkan is enabled, the C++ pipeline loads Vulkan FP16 Part1/2/3 and batch-2 Part1/2 if present; otherwise it falls back to INT8 single-patch. Part4 (chunked) is unchanged. No separate “Vulkan” toggle; backend is determined by which `.pte` files are present and the runtime flag.

5. **Driver workarounds (if Vulkan still crashes)** — Try on the device (requires root or dev options):
   - `adb shell settings put global debug.vulkan.no_validation 1`
   - `adb shell setprop debug.vulkan.version 1.1`  
   (Some INT8 staging bugs are reduced on Vulkan 1.1. FP16 B2 remains the most reliable.)

6. **AOT tensor safety (C++)** — To avoid Vulkan AOT metadata corruption affecting the JNI return, the Part4b single path validates the output tensor (dim==3, numel % 14 == 0, N &lt; 2M) and copies it into a local buffer before pruning and returning. No custom AOT arena APIs are used; VulkanPartitioner’s default AOT planning remains in effect.

7. **Runtime failure: BackendFailed (error 32 / 0x20)** — If the app logs `Part1 fail 1x (0,0): forward error 32` or `AFTER_PART1_FORWARD forward_error=32`, the **Vulkan delegate failed during execution** (not at load). This is usually due to incompatible shader ops, driver bugs, or device-specific Vulkan limits rather than OOM (e.g. `avail=2172MB` is typically sufficient).
   - **Input size note:** Part1 receives **384×384 patches** (cropped from the 1536×1536 image), not the full resolution. If you still hit limits, reduce capture resolution or use a smaller pipeline input.
   - **Verify delegate export:** Ensure Part1/Part2 were built with the Vulkan partitioner. Run `python inspect_pte_delegates.py <path/to/sharp_split_part1_vulkan_fp16.pte>` to confirm "VERIFIED: Vulkan delegate is present".
   - **Verify device requirements:** Ensure the device supports the Vulkan version required by the ExecuTorch runtime; see [Ultralytics checks reference](https://docs.ultralytics.com/reference/utils/checks/) for requirement checks.
   - **Reduce memory pressure:** If you must use Vulkan, try a smaller input size (e.g. lower capture resolution) as in [Model Size Optimization](https://docs.ultralytics.com/integrations/executorch/#model-size-optimization).
   - **Reference:** [Ultralytics ExecuTorch guide](https://docs.ultralytics.com/integrations/executorch/); [Ultralytics GitHub Issues](https://github.com/ultralytics/ultralytics/issues) for Android Vulkan compatibility; op-level compatibility is in `EXECUTORCH_VULKAN_OPS.md`.
