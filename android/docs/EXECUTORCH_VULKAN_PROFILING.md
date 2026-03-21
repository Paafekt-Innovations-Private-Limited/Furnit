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

## 3. Concrete profiling plan for slow Part1+2

This is the concrete workflow to use when Part1+2 is the long pole.

### One-tap in-app investigation

The Android app now has a single in-app action for the three checks above:

- `Profile -> Settings -> Developer -> Part1 only test -> Investigate all 3`

That action logs:

- the current room-routing decision
- forced Vulkan Part1 `3x` on the same patch
- forced CPU-sidecar Part1 `3x` on the same patch
- exact `executor_runner` / ETDump commands for standalone Part1 and Part2 artifacts

Use these grep markers after running it:

```bash
adb logcat -d | grep P1_ROUTE
adb logcat -d | grep P1_INVESTIGATE
adb logcat -d | grep P12_ETDUMP
```

Read them this way:

- `P1_ROUTE`: proves whether normal room creation is effectively hybrid or true Vulkan for Part1+2
- `P1_INVESTIGATE`: lists which candidate `.pte` files were found and prints timed `3x` forwards for forced Vulkan and forced CPU-sidecar paths
- `P12_ETDUMP`: prints ready-to-run shell commands for standalone ETDump collection

### Step 0: first verify which backend you are actually measuring

Before doing any Vulkan diagnosis, inspect the room-creation log line:

```text
runFullPipelineInt8: modelDir=... useVulkan=1 part12OnCpu=...
```

Read it this way:

- `part12OnCpu=1` means the room run is **hybrid**, not Vulkan Part1+2.
- `part12OnCpu=0` means the room run is attempting true Vulkan Part1+2.

This matters because a `30s+` Part1+2 block in a hybrid run is **not evidence of a Vulkan Part1 bottleneck**.

In the current app, hybrid Part1+2 can be auto-enabled when CPU Part1/2 sidecar models exist under `models_cpu`. So if you want to measure true Vulkan Part1+2 in the room pipeline, make sure the run actually logs `part12OnCpu=0`.

### Step 1: measure standalone Part1 first inside the app

Use the existing in-app Part1 benchmark path before going deeper into ETDump.

In `Profile -> Settings -> Developer`:

1. Tap `Release Part1 cache`.
2. Tap `Warmup`.
3. Run `Benchmark 3×`.

Capture:

```bash
adb logcat -d | grep P1_BENCH
adb logcat -d | grep PART1_RUN
adb logcat -d | grep PART1_ARTIFACT
```

What this gives you:

- `P1_BENCH session_ensure_ms=...` shows load + 2x warmup if the session was cold.
- `P1_BENCH timed_forward 1/3`, `2/3`, `3/3` shows steady forward time on the same module and same patch.
- `PART1_ARTIFACT` confirms which `.pte` file was actually used.

Interpretation:

- `run1` much larger than `run2` / `run3` means pipeline/shader setup dominates the first call.
- all three runs similarly high means the real cost is steady-state execution.

### Step 2: get the portable CPU baseline

Run the same Part1 benchmark using the portable / CPU artifact and compare it to the Vulkan artifact on the same device.

For Part1 diagnosis, the most useful comparison is:

- Vulkan Part1: `sharp_split_part1_vulkan_fp32.pte` or `_fp16.pte`
- Portable Part1: `sharp_split_part1.pte`

Why this matters:

- If Vulkan is only slightly better or worse than portable, the delegated graph may be suffering from layout churn or graph breaks.
- If Vulkan is much faster on the same patch, then room-level slowness may be elsewhere.

### Step 3: collect ETDump for standalone Part1 and Part2

Once you know a standalone forward is still slow, use `executor_runner` + ETDump on the individual artifacts.

Part1:

```bash
adb shell mkdir -p /data/local/tmp/etvk/models /data/local/tmp/etvk/etdumps
adb push sharp_split_part1_vulkan_fp32.pte /data/local/tmp/etvk/models/
adb shell /data/local/tmp/etvk/executor_runner \
  --model_path /data/local/tmp/etvk/models/sharp_split_part1_vulkan_fp32.pte \
  --num_executions=3 \
  --etdump_path /data/local/tmp/etvk/etdumps/part1_vulkan.etdp
adb pull /data/local/tmp/etvk/etdumps/part1_vulkan.etdp ./
python devtools/inspector/inspector_cli.py --etdump_path part1_vulkan.etdp
```

Part2:

```bash
adb push sharp_split_part2_vulkan_fp32.pte /data/local/tmp/etvk/models/
adb shell /data/local/tmp/etvk/executor_runner \
  --model_path /data/local/tmp/etvk/models/sharp_split_part2_vulkan_fp32.pte \
  --num_executions=3 \
  --etdump_path /data/local/tmp/etvk/etdumps/part2_vulkan.etdp
adb pull /data/local/tmp/etvk/etdumps/part2_vulkan.etdp ./
python devtools/inspector/inspector_cli.py --etdump_path part2_vulkan.etdp
```

Also run the portable equivalents for baseline if needed.

### Step 4: read the Inspector output the right way

The main question is whether time is inside the delegate’s real math kernels or in graph/layout overhead.

#### Case A: `DELEGATE_CALL` is almost the whole `Method::execute`

This means most time is inside the delegated region itself.

Then inspect the expensive ops:

- If a few large ops dominate, such as `bmm`, `softmax`, `linear`, or other attention-heavy math, the next lever is usually **splitting / repartitioning around the attention-heavy blocks**.
- If the largest ops are conv-heavy and stable, the next lever is usually **graph-specific export work**, not app-side code.

#### Case B: many small conversion / layout ops show up

This usually means layout or storage churn is stealing time.

Common warning signs are many ops with names like:

- `view_*`
- `permute_*`
- `concat_*`
- `image_to_nchw_*`
- `nchw_to_image_*`
- `texture3d_*`

If the dump looks like that, the next lever is usually:

- fewer delegated regions
- re-export with different partitioning
- split the model around the transition-heavy area

This matches the official Vulkan troubleshooting guidance: poor performance often comes from extra copies inserted to satisfy layout/storage requirements.

#### Case C: lots of graph breaks / CPU fallback

If the profile or logs suggest unsupported ops are forcing frequent exits from Vulkan, the next lever is:

- reduce the Vulkan partition size
- change the export/lowering so unsupported boundaries are cleaner
- test whether a smaller delegated region is actually faster than “maximum Vulkan”

### Step 5: what not to expect from current Vulkan

Do not assume the fix will be “turn on quantized Vulkan convs”.

Per the official Vulkan backend overview:

- Vulkan already supports quantized **linear** paths
- broader quantized operator support such as quantized **convolution** is still evolving

So for SHARP Part1, a large win is more likely to come from:

- reducing layout churn
- reducing graph-break overhead
- splitting around attention-heavy regions

not from a generic switch that suddenly makes all Part1 convolutions fast on Vulkan.

### Step 6: decision rule after profiling

Use this decision table:

- standalone Part1 Vulkan slow, ETDump dominated by `view` / `concat` / `texture3d` conversions:
  - prioritize repartitioning and export-side layout cleanup
- standalone Part1 Vulkan slow, ETDump dominated by a few big attention/math ops:
  - prioritize Part1a / Part1b split experiments around attention-heavy blocks
- standalone Part1 Vulkan okay, but room pipeline still slow:
  - the long pole is probably not standalone Part1 math; check Part2, room-level routing, or hybrid CPU behavior
- room log shows `part12OnCpu=1`:
  - do not treat that run as evidence about Vulkan Part1 performance

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
