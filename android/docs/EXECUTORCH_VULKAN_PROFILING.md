# ExecuTorch Vulkan profiling for SHARP ML room creation

This doc describes how to profile the SHARP ExecuTorch Vulkan pipeline (Part1–Part4b) on Android, following the [ExecuTorch Vulkan profiling tutorial](https://docs.pytorch.org/executorch/1.0/backends/vulkan/tutorials/etvk-profiling-tutorial.html).

## 1. Vulkan GPU profiling in logcat (no rebuild)

Enable the ExecuTorch Vulkan runtime’s built-in profiling so logcat shows where time is spent (GPU dispatches, CPU fallback stalls):

```bash
# From repo android/
./enable_vulkan_profiling.sh
```

Then run the app, create a room (SHARP ML path), and inspect logcat. To turn off:

```bash
adb shell setprop debug.executorch.vulkan.enable_profiling 0
```

## 2. ETDump + Inspector (per-op and delegate timing)

For per-operator and delegate-level timing (e.g. `DELEGATE_CALL`, `Method::execute`, and individual Vulkan kernels), you need an **ETDump** and the **Inspector** CLI.

### Option A: In-app ETDump (Part4b only)

When ExecuTorch is built with event tracer and devtools, the app can write an ETDump for the **Part4b** Vulkan run to app storage.

**Note:** The Maven AAR’s `libexecutorch.so` is built *without* devtools, so `ETDumpGen` is not linked. The app is built with ETDump code **disabled** by default so it links against the AAR. To use in-app ETDump you must use a **local** ExecuTorch lib built with devtools and build with `-PexecutorchUseLocalLib` (which turns on `EXECUTORCH_PROFILING_ETDUMP` in CMake).

1. **Build ExecuTorch with event tracer and devtools** (so the app’s native lib can record ETDump):
   ```bash
   cd android
   ./build_executorch_vulkan_for_furnit.sh
   ```
   This script now sets `EXECUTORCH_ENABLE_EVENT_TRACER=ON` and `EXECUTORCH_BUILD_DEVTOOLS=ON`.

2. **Copy etdump headers** so the app’s CMake can see them (optional; only if you want in-app ETDump):
   ```bash
   # Ensure third_party/executorch has devtools/etdump (from ExecuTorch clone)
   ./scripts/setup_executorch_headers.sh
   ```
   If your clone has `devtools/etdump`, the setup script copies it and CMake will define `EXECUTORCH_HAS_ETDUMP` for `sharp_executorch_full`.

3. **Build the app** with the local ExecuTorch lib:
   ```bash
   ./gradlew assembleEtVulkanDebug -PexecutorchUseLocalLib
   ```

4. **Enable “Record ETDump on next room creation”** in Settings (under the ExecuTorch / Vulkan section).

5. **Create a room** (SHARP ML path). The app writes `sharp_part4b.etdp` to app external files (e.g. `.../files/sharp_part4b.etdp`).

6. **Pull and inspect**:
   ```bash
   adb pull /data/data/com.furnit.android/files/sharp_part4b.etdp ./
   # From an ExecuTorch repo with devtools:
   python devtools/inspector/inspector_cli.py --etdump_path sharp_part4b.etdp
   ```

### Option B: executor_runner (full model, no app)

To profile a **single .pte** (e.g. one Part4b or Part1 model) without the app:

1. Build ExecuTorch with `EXECUTORCH_ENABLE_EVENT_TRACER=ON` and `EXECUTORCH_BUILD_EXECUTOR_RUNNER=ON`, then build the runner (see the [tutorial](https://docs.pytorch.org/executorch/1.0/backends/vulkan/tutorials/etvk-profiling-tutorial.html)).

2. Push the .pte and runner to the device, then run with `--etdump_path` and optional `--num_executions`:
   ```bash
   adb shell mkdir -p /data/local/tmp/etvk/etdumps/
   adb shell /data/local/tmp/etvk/executor_runner \
     --model_path /data/local/tmp/etvk/models/sharp_split_part4b_vulkan.pte \
     --num_executions=3 \
     --etdump_path /data/local/tmp/etvk/etdumps/part4b.etdp
   adb pull /data/local/tmp/etvk/etdumps/part4b.etdp ./
   python devtools/inspector/inspector_cli.py --etdump_path part4b.etdp
   ```

## Inspector output

The Inspector prints a table with columns such as:

- **Event** (e.g. `Execute`, `DELEGATE_CALL`, `Method::execute`)
- **Block / name** (e.g. `conv2d_clamp_half_163`, `linear_naive_texture3d_half_174`)
- **Min / Max / Mean** times (ms) across runs

Use this to find slow Vulkan kernels or CPU fallbacks and to compare devices or builds.

## Slow performance (troubleshooting)

Per the [Vulkan backend troubleshooting](https://docs.pytorch.org/executorch/1.0/backends/vulkan/vulkan-troubleshooting.html) doc, slowness is often due to:

- **Key compute shaders** (e.g. convolution or linear) not performing well on your GPU
- **Unsupported operators** causing too many graph breaks
- **Memory layout / storage** forcing extra copies between ops

**What to do:**

1. **Obtain profiling data** (this is what the troubleshooting page asks for):
   - **Logcat:** Run `./enable_vulkan_profiling.sh`, then create a room and capture logcat. On zsh, quote the tags so `*` is not globbed (e.g. `adb logcat -s 'executorch:*' 'sharp_executorch_full:*' | tee logcat_vulkan.txt`, or `adb logcat -d | tee logcat_vulkan.txt` for a one-shot dump). To reduce noise, use only `sharp_executorch_full`: `adb logcat -v threadtime -s 'sharp_executorch_full:*' | tee logcat_vulkan.txt`.
   - **ETDump + Inspector:** Use “Record ETDump on next room creation” (or executor_runner with `--etdump_path`), pull the `.etdp`, run `inspector_cli.py --etdump_path <file>.etdp`, and save the table output.

2. **Optional – file an issue** on [ExecuTorch GitHub](https://github.com/pytorch/executorch) with:
   - Device(s) tested and which ones are slow
   - The profiling data (logcat snippet and/or Inspector output)
   - ExecuTorch version (e.g. 1.1.0) or commit hash if built from source
   - If possible: export script and the `.pte` (or at least which SHARP parts: Part1/2/3/4b)

Using the steps above (Section 1 and Section 2) gives you the data the troubleshooting guide expects.

## References

- [ExecuTorch Vulkan profiling tutorial](https://docs.pytorch.org/executorch/1.0/backends/vulkan/tutorials/etvk-profiling-tutorial.html)
- [Vulkan backend troubleshooting (slow performance)](https://docs.pytorch.org/executorch/1.0/backends/vulkan/vulkan-troubleshooting.html)
- [ETDump (ExecuTorch Dump)](https://docs.pytorch.org/executorch/stable/etdump.html)
- [Model Inspector](https://docs.pytorch.org/executorch/stable/model-inspector.html)
