# ExecuTorch Vulkan Backend Registration

## Problem

When running the pipeline with **ExecuTorch INT8 (Vulkan)** selected, logcat shows:

```
ExecuTorch: Backend VulkanBackend is not registered.
sharp_executorch_full: Part1 fail 1x (0,0): forward error 32 BackendFailed(0x20).
```

The default **executorch-android** AAR (Maven `org.pytorch:executorch-android:1.1.0`) is built with the **XNNPACK** backend. It does **not** include the Vulkan delegate, so the runtime never registers `VulkanBackend`. Any `.pte` file that was exported with Vulkan delegation then fails at forward time.

## Fix: Link the Vulkan-built ExecuTorch lib

ExecuTorch ships a separate AAR that includes the Vulkan backend: **executorch-android-vulkan**. To register the Vulkan backend you must link (or load) the native library from that AAR instead of the default one.

### Option 1: Build with Vulkan AAR’s lib (recommended for Vulkan path)

1. Clean and extract the Vulkan AAR’s `.so` into `executorch_lib`:
   ```bash
   cd android
   ./gradlew clean
   ./gradlew extractExecutorchSoFromAar -PexecutorchUseVulkanLib
   ```
2. Rebuild the app so `libsharp_executorch_full.so` links against the new `libexecutorch.so`:
   ```bash
   ./gradlew assembleDebug
   ```

The `-PexecutorchUseVulkanLib` property makes the extract task take `org.pytorch:executorch-android-vulkan:1.1.0` and copy its `jni/arm64-v8a/*.so` into `app/src/main/cpp/executorch_lib/` as `libexecutorch.so`. CMake then links `sharp_executorch_full` against this Vulkan-enabled lib, so the Vulkan backend is registered at load time.

**Load order (Java `Module` path):** The app calls `ExecutorchNativeLoader.loadForJavaModule()` before `Module.load()`: `libexecutorch_core.so` (if present), then `libexecutorch.so` (Vulkan backends register via static init here), then `libexecutorch_jni.so`. Loading JNI before the core runtime can leave backends unregistered.

**Startup warmup:** `FurnitApplication` calls `ExecutorchNativeLoader.loadForJavaModule()` in `onCreate`, then `Part1OnlyTest.scheduleStartupWarmup()` (background thread). If `sharp_split_part1.pte` and `part1_test_patch_f32.bin` are on device, one Part1 `forward()` runs during app launch so Vulkan pipeline/shader work can finish before the user runs inference. Logcat: `adb logcat -s ExecuTorchWarmup:I`.

**Settings manual warmup:** Developer → **Part1 warmup** → **Warmup**. Status line on screen reads `furnit_prefs` (`part1_warmup_state`, duration ms). Machine-readable log lines use tag **`Part1Warmup`** with prefix `WARMUP_STATUS`:
```bash
adb logcat -s Part1Warmup:I
adb logcat -d | grep WARMUP_STATUS
```

**Testing registration:** Run **Settings → Developer → Part1 only test → Run**. If the Part1 .pte uses Vulkan and the backend is not registered, the test fails with a clear message (“Vulkan not registered. Build with Vulkan AAR”) and logcat tag `Part1Test`. On success it logs “Vulkan registration: OK”.

**PushConstantData.h:** If you built or patched ExecuTorch and changed `backends/vulkan/runtime/graph/containers/PushConstantData.h` line 19 (`kMaxPushConstantSize`) from 128 to 256, revert it to 128; that file is in the ExecuTorch repo, not Furnit.

**Note:** The Vulkan AAR may omit some portable/XNNPACK ops. If the **CPU ExecuTorch INT8** path (portable/INT8 .pte) starts failing with error 0x14 or missing ops, you are likely hitting that limitation; use the default AAR (no flag) for CPU-only, or build ExecuTorch from source with both XNNPACK and Vulkan.

**Python verification:** To run Vulkan-exported `.pte` in Python (e.g. `verify_export_part1_part2.py`), the executorch package must be installed/built **with Vulkan support** so that VulkanBackend is registered in the Python runtime. The default `pip install executorch` typically does not include it. See [EXECUTORCH_EXPORT_VERIFY.md](EXECUTORCH_EXPORT_VERIFY.md) for options (portable-only verification or building executorch with Vulkan for the host).

### Option 2: Build ExecuTorch from source with full Vulkan kernels

When the prebuilt Vulkan AAR causes **slow Part1 (~16s per patch) and/or a crash after AFTER_PART1_FORWARD**, the AAR’s limited/curated shader set or driver behavior is the cause. The only reliable fix is to build ExecuTorch from source with **full Vulkan kernels** (all GLSL shaders compiled in, not a pruned set).

See **[EXECUTORCH_BUILD_VULKAN_FROM_SOURCE.md](EXECUTORCH_BUILD_VULKAN_FROM_SOURCE.md)** for step-by-step: clone ExecuTorch, set `EXECUTORCH_BUILD_VULKAN=ON`, run `scripts/build_android_library.sh` (with Vulkan SDK `glslc` in PATH), then copy the built `libexecutorch.so` into this app’s `executorch_lib/` and build with `-PexecutorchUseLocalLib` so the extract task does not overwrite it.

## VK_ERROR_DEVICE_LOST (Mali / Pixel GPU timeout)

If logcat shows **Vulkan Fence: Device lost** or **VK_ERROR_DEVICE_LOST** at `Fence.cpp` during Part1 forward, the GPU driver has reset mid-submission (TDR). This is a **known Mali-G715 driver bug** on Pixel devices (Android 15+) with heavy Vulkan compute in ExecuTorch—stock Mali drivers can lose the device during prolonged compute (e.g. 35× 384×384 Part1 patches, or ops around downsample).

### In-app workaround

In Settings, enable **Part1+2 on CPU (when Vulkan)**. Part1 and Part2 then run on CPU (portable models); Part4a and Part4b stay on Vulkan. The model dir must contain both portable Part1+2 (e.g. `sharp_split_part1_int8.pte`, `sharp_split_part2_int8.pte`) and the Vulkan Part4* files. This option is **off by default** (Part1+2 use Vulkan by default).

### Confirmed fixes (device / workload)

- **PanVK (Mesa)**: Install Mesa’s open-source PanVK driver for Mali Valhall (G715) via Magisk/root. Vulkan 1.2 conformant (as of 2026) and fixes device-lost on Pixels in ExecuTorch-style workloads. Search e.g. “PanVK Magisk Pixel 9a”.
- **Firmware**: Ensure latest Pixel OTA; Google has patched some Mali fence behaviour in recent builds.
- **Reduce workload**: Use **1280×1280** in Settings (ExecuTorch INT8) to lower resolution; or bisect/export Part1 without heavy downsample in the graph. Our C++ downsample (2x/4x) is already CPU/NEON, but the Part1 Vulkan graph may still hit driver limits.
- **Validate Vulkan**: `adb shell vulkaninfo | grep deviceName` and check extensions; missing `VK_KHR_synchronization2` can contribute to losses.

## Part1 test modes (fewer / minimal patches)

In Settings, two options reduce Part1 workload for fence/timeout debugging:

- **Part1 test: fewer patches (5+2+1)** — runs 8 patches instead of 35 (5 at 1x, 2 at 0.5x, 1 at 0.25x).
- **Part1 test: minimal (1+0+1)** — runs only 2 Part1 forwards (1 at 1x, 0 at 0.5x, 1 at 0.25x). Use to see if the crash is tied to patch count or happens even with minimal work. Output is invalid.

To capture the actual crash (backtrace and signal), run logcat without filtering, or after reproducing run:
`adb logcat -d | grep -E "Fatal signal|libc\+\+abi|DEBUG|backtrace|sharp_executorch"` and share the lines around the crash.

## Vulkan & ExecuTorch diagnostics (shaders, sync, device)

In **Settings** → **Vulkan & ExecuTorch diagnostics** → tap **Run**. The app logs:

- **Vulkan device:** deviceName, driverVersion, apiVersion, vendorID/deviceID
- **Device extensions:** full list (and whether **VK_KHR_synchronization2** / **VK_KHR_timeline_semaphore** are present; missing sync2 can contribute to fence/device-lost)
- **Shader note:** The shader registry is inside the ExecuTorch native lib and is not enumerable from the app; build from source with `EXECUTORCH_BUILD_VULKAN=ON` for full shaders.

**To see the diagnostic output**, run (device connected via USB):

```bash
adb logcat -s VulkanDiag:D Vulkan1536Test:D
```

**If Sync shows VK_KHR_synchronization2=YES and you still get device-lost or crash during Part1:** the driver exposes sync2, so the cause is not “missing sync extension”. It is then likely **GPU timeout (TDR)** on Mali for long-running Part1 compute, or the **prebuilt ExecuTorch Vulkan AAR** (limited shaders / sync paths). Use **Part1+2 on CPU** as workaround, or build ExecuTorch from source with full Vulkan.

Clear logcat first then tap Run, or run the command and then tap **Run** in Settings to see the lines appear. For a one-shot capture after tapping Run:

```bash
adb logcat -d -s VulkanDiag:D Vulkan1536Test:D
```

## Crash during Part1 forward (last log = BEFORE_PART1_FORWARD)

If logcat shows **BEFORE_PART1_FORWARD** (and optionally "calling m1->forward") but **never** "AFTER_PART1_FORWARD":

- **Crash location:** The process dies **inside** the Vulkan Part1 forward — i.e. during the GPU compute for the first 384×384 patch. This is typically **GPU timeout / device-lost** (Mali driver) or a Vulkan dispatch crash, not readback.
- **Workaround:** In Settings, enable **Part1+2 on CPU (when Vulkan)**. Part1 and Part2 then run on CPU (portable .pte); Part4 stays on Vulkan. The pipeline will complete.

## Slow Part1 (~16s) and crash after AFTER_PART1_FORWARD

If logcat shows Vulkan Part1 completing successfully but taking ~16 seconds per patch, followed by **PROCESS ENDED** (no Fatal signal in log):

- **Crash location:** The process dies immediately after `AFTER_PART1_FORWARD 1x (0,0) status=ok`. The next step in code is reading Part1’s output tensors (`const_data_ptr<float>()`), which triggers a **Vulkan sync** (GPU→CPU readback). The crash is almost certainly either (1) during that readback (fence wait / device-lost) or (2) at Part2’s first forward.
- **Workaround:** In Settings, enable **Part1+2 on CPU (when Vulkan)**. Part1 and Part2 then run on CPU (portable .pte); Part4 stays on Vulkan. The pipeline will complete; only Part1+2 are moved off GPU.

If logcat shows Vulkan Part1 completing successfully but taking ~16 seconds per patch, followed by a crash (SIGABRT or missing shader):

- **Cause:** The prebuilt **executorch-android-vulkan** AAR is built with a limited set of Vulkan shaders and/or unoptimized paths. SHARP’s transformer ops may hit missing shaders, fallbacks, or driver timeouts.
- **Fix:** Build ExecuTorch from source with `EXECUTORCH_BUILD_VULKAN=ON` so the Vulkan backend is compiled with the full in-tree GLSL compute shader library. No “shader pre-registration” is needed in the app—linking the Vulkan-built lib registers the backend and its shaders via static initialization.
- **C++ / ShaderRegistry:** The app cannot add shaders at runtime; the ShaderRegistry is compiled into the ExecuTorch native lib. Use a lib built from source with full kernels.

## Missing shader crash (view_convert_buffer_float_half)

After fixing registration, Vulkan may still crash with:

```
libc++abi: terminating due to uncaught exception of type vkcompute::vkapi::Error: ...
Could not find ShaderInfo with name view_convert_buffer_float_half
Fatal signal 6 (SIGABRT)
```

**Cause:** The prebuilt **executorch-android-vulkan** AAR is built with a limited set of Vulkan shaders. The SHARP Part1 (or other) Vulkan .pte uses a float/half conversion that requires the `view_convert_buffer_float_half` shader, which is not included in the AAR's shader registry.

**Workarounds:**

1. **Use CPU path:** In Settings, select **CPU ExecuTorch INT8** and use the INT8 + tile_00 model set (Part1/2/3 int8, Part4a 512/65, Part4b_tile_00). This path is stable and does not depend on Vulkan shaders.
2. **Build ExecuTorch from source with Vulkan:** Build the Android lib with `EXECUTORCH_BUILD_VULKAN=ON` and ensure the Vulkan backend is built with all required shaders (including float/half conversions). Replace the app's `executorch_lib/libexecutorch.so` with that build.
3. **Re-export the model for AAR shaders:** Export SHARP with Vulkan using the same ExecuTorch version as the AAR and the **`--vulkan-aar-compat`** option so the .pte only uses shaders present in the prebuilt AAR:
   ```bash
   cd android
   python export_sharp_executorch_split4.py --backend vulkan --chunked-part4 --vulkan-aar-compat --output-dir sharp_vulkan_aar_compat
   ```
   This uses `VulkanPartitioner(compile_options={"force_fp16": False, "skip_memory_planning": False})` and FP32 dtype, avoiding the `view_convert_buffer_float_half` shader. Install ExecuTorch from source or pip to match the AAR version (e.g. 1.1.0) before exporting.

**Build variants:** Use Gradle flavors **etVulkan** (Vulkan AAR) vs **etCpu** (XNNPACK AAR). Examples:  
`./gradlew :app:assembleEtVulkanDebug` and `./gradlew :app:assembleEtCpuDebug`.  
`assembleDebug` builds **both** flavors (two APKs). See `android/docs/TEST_INT8_IN_APP.md`.

When using a **local** ExecuTorch build (from source with full Vulkan), copy your `libexecutorch.so` into `app/src/main/cpp/executorch_lib_etVulkan/` (or `executorch_lib_etCpu/` if testing that flavor) and build with `-PexecutorchUseLocalLib` so the extract task does not overwrite it.

## Logcat: mem-pressure-event and service restarts

If you see many lines like:

```
ActivityManager: Rescheduling restart of crashed service ... in ...ms for mem-pressure-event
ActivityManager: Scheduling restart of crashed service ... for connection
```

**Meaning:** The system is under **memory pressure**. The kernel or LMK has killed one or more processes (often system/Google services: GMS, GcmService, AiAi, iwlan, etc.). ActivityManager is deferring their restart by a long delay (e.g. 1.5M ms ≈ 25 min) so that memory can settle. Later lines with `in 0ms for mem-pressure-event` mean the system is still in a pressure state and keeps postponing restarts.

**Relation to Furnit:** Heavy inference (Part1+2 Vulkan, 35 patches, 1536×1536) can push total device memory into the pressure zone. The system may then kill Furnit (SIGKILL, signal 9), other apps, or system services. So **mem-pressure-event** in logcat confirms **OOM / low-memory kills**, not necessarily a Vulkan device-lost or GPU crash.

**Mitigations:**

- **Part1+2 on CPU (when Vulkan):** Frees GPU memory and reduces peak (Settings).
- **Part1+2 in chunks (yield between):** Processes patches in smaller chunks with 50 ms sleep between chunks to ease GPU/RAM pressure (Settings).
- **Part1 test: fewer patches (5+2+1)** or **minimal (1+0+1):** Fewer Part1 forwards for testing (Settings).
- **Close other apps** before a long scan to free RAM.
- **onTrimMemory:** The app already registers `onTrimMemory` (FurnitApplication) to release native caches on `TRIM_MEMORY_RUNNING_*` / `TRIM_MEMORY_COMPLETE`; this helps when the system is reclaiming memory.

See [MEMORY_OPTIMIZATION.md](MEMORY_OPTIMIZATION.md) for Part 4 decoder memory and chunked Part 4.

## References

- [Vulkan Backend — ExecuTorch 1.0](https://docs.pytorch.org/executorch/1.0/android-vulkan.html): Vulkan AAR and linking with `--whole-archive` when building from source.
- [Using ExecuTorch on Android](https://docs.pytorch.org/executorch/stable/using-executorch-android.html): AAR by backend (XNNPACK vs Vulkan), and “Using Vulkan Backend” build option.
