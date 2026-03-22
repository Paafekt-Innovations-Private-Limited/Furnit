# Known-Good ExecuTorch Vulkan Room Creation Flow

This document records the current working Vulkan room-creation path for SHARP in Furnit.

Date of the reference run: March 21, 2026.

This is the flow that currently gives:

- stable room creation on the `etVulkan` APK
- the optimized `tile_00` Part4 path
- visually strong room output on a real indoor capture

## Build and model set

- APK flavor: `etVulkan`
- Models directory on device: `files/models_cpuvulkan_hybrid`
- Active Part4 route:
  - `stage_pre` on Vulkan
  - `decoder_head` on Vulkan
  - `init_base` on portable
  - `raw_heads` on Vulkan
  - `compose` on portable
- Active optimized model variant: `part4_vulkan_hotpath_lite_v1`
- Hot-path surgery:
  - grouped-conv simplification enabled
  - groups: `4`
  - modified hot-path layers: `43`
  - estimated touched hot-path parameter reduction: `74.95%`

## Part4 technical problem solved

The Part4 work solved **two different technical problems**, not just a generic “decoder was slow” issue.

### 1. Vulkan-unsafe shape path in old Part4b routing

The older Vulkan Part4b paths were brittle because some late decoder / packing paths created tensor shapes that the Vulkan delegate did not handle safely.

The clearest bad case in this repo is the legacy `tile_full` path:

- Vulkan graph init could abort inside `libexecutorch.so` with `sizes_.size() <= 4` failures from `Tensor.cpp` during concat-related lowering.
- That means the delegate/lowered graph hit tensor-size metadata outside the backend’s expected shape envelope.
- This was a Vulkan backend / lowered-graph problem, not a Furnit JNI problem.

That is why the current working route does **not** try to keep all of Part4b inside one monolithic Vulkan graph.

### 2. The real hot path was concentrated in `decoder_head`

Even after moving away from the unsafe monolithic route, the remaining tiled Part4 path was still too slow because one subgraph dominated tile latency:

- `stage_pre` was relatively small
- `init_base` and `compose` were small
- `decoder_head` was the heavy conv/upsample/fusion block and the main per-tile cost center
- `raw_heads` was the second major cost center

The logs from the current known-good run still show that shape clearly:

- `stage_pre`: about `120-127ms`
- `decoder_head`: about `780-788ms`
- `raw_heads`: about `418-427ms`
- `compose`: about `51-69ms`

### What the fix actually was

The current Part4 solution is a **fine split `tile_00` route**, not a single-model “optimize everything” export.

The split does this on purpose:

- `stage_pre` stays on Vulkan
- `decoder_head` stays on Vulkan
- `raw_heads` stays on Vulkan
- `init_base` stays portable
- `compose` stays portable

The reason `init_base` and `compose` stay portable is that these paths create **rank-5 tensors** / shape-sensitive packing logic that are not safe to push through the current Vulkan delegate path in this app. The split keeps Vulkan on the conv-heavy parts and keeps the shape-fragile packing stages off Vulkan.

The native C++ tiled route also does explicit stage-to-stage handoff:

- each stage output is copied into owned CPU buffers
- expected shapes are validated before the next stage runs
- the next stage is invoked from fresh `from_blob(...)` tensors

That avoids relying on fragile cross-stage reuse of delegate-owned tensors across Vulkan and portable subgraphs.

### Hot-path-lite optimization on top of the split

Once the route was stable, the heavy `decoder_head` hot path was optimized directly:

- grouped-conv simplification was applied to the conv-heavy decoder hot path
- `43` hot-path layers were modified
- the hot-path surgery used `groups=4`
- the estimated touched hot-path parameter reduction was `74.95%`

This did not change the overall room-creation routing. It reduced compute inside the hottest Part4 tile path.

### Net result

So the Part4 solution was:

1. stop trying to run the wrong Part4b shape path as one Vulkan blob
2. split the tile route around the Vulkan-unsafe tensor-shape boundary
3. keep only the conv-heavy subgraphs on Vulkan
4. then optimize the real hotspot (`decoder_head`) inside that stable route

That is why the current flow is both:

- stable enough to finish
- much faster on Part4 than the older route
- still visually strong, because the final packing / compose logic stayed on the safer path

## Settings to use

Open `Profile -> Settings -> Developer` and use:

| Setting | Value |
|---|---|
| `Inference Backend` | `ExecuTorch INT8 (Vulkan)` |
| `Max Gaussians` | `All` |
| `Use true 1280x1280` | `Fixed OFF` |
| `Prefer Vulkan FP16 models` | `Fixed ON` |
| `Prefer single Part4b` | `Fixed OFF` |
| `Record ETDump on next room creation` | `OFF` for normal runs |

Notes:

- `Prefer single Part4b = OFF` is intentional. The current known-good Vulkan path is the fine-split `tile_00` route, not the old single-decoder route.
- `Use true 1280x1280 = OFF` keeps the currently validated `1536` hybrid split path.
- Those three values are now fixed in the Android app and hidden from the Settings UI.
- That cleanup did not modify ExecuTorch itself; it was limited to app-layer settings handling.

## Working flow

1. Build and install the `etVulkan` APK.
2. Push the Vulkan SHARP model pack to `models_cpuvulkan_hybrid`.
3. In Settings, select the backend and confirm `Max Gaussians`. The three hidden ExecuTorch values above are already fixed by the app.
4. Create a room from a single photo as usual in the app.
5. Let the pipeline complete without changing backend toggles mid-run.

## What the logs should show

The important routing lines are:

```text
runFullPipelineInt8: modelDir=.../models_cpuvulkan_hybrid useVulkan=1 ... preferSingleP4b=0
runPart4bBatchedTiledPipeline: fine split tile_00 artifacts present, preferring sequential tile_00 path
runPart4bTiledFullPipeline: using fine split tile_00 path ...
runPart4bTiledFullPipeline: loaded fine split tile_00 modules
```

For a successful run, the tail should include:

```text
[TIMING] Part4b tiled (full pipeline C++): 24275ms, 1179648 Gaussians
JNI RETURN: size=16515072 validated (tiled)
```

## Measured timings from the reference run

End-to-end room creation:

- start: `08:44:04.253`
- validated native return: `08:45:32.655`
- total: about `88.4s`

Major stage timings:

- Part1 + Part2 patches: `31.253s`
- Part3: `16.581s`
- Part4a total: `16.138s`
- Part4b tiled full pipeline: `24.275s`

Steady-state Part4b per-tile timings from the same run:

- `stage_pre`: about `120-127ms`
- `decoder_head`: about `780-788ms`
- `init_base`: about `11-19ms`
- `raw_heads`: about `418-427ms`
- `compose`: about `51-69ms`
- tiles `2-16`: about `1.45-1.47s` each

## March 21, 2026 runtime cleanup note

A small app-native Part1+2 cleanup was applied after the main Vulkan path was already working:

- skip `0.5x` / `0.25x` image downsample prep when the app is in the fixed `part12_25_only` room-creation path
- remove extra steady-state per-patch Part1+2 debug logs

This solved wasted runtime work in the encoder path. It did **not** change ExecuTorch, model exports, or the Part1/Part2 graph itself.

From the later verified run with this cleanup:

- `runFullPipelineInt8` start: `10:17:53.658`
- `JNI RETURN` validated: `10:19:20.625`
- total: `86.967s`
- Part1+2: `30.478s`
- Part3: `15.982s`
- Part4a total: `16.109s`
- Part4b tiled: `24.250s`

That later run also shows `part12OnCpu=1`, so the current verified flow was still **hybrid** for Part1+2.

## Performance benefit

Before the hot-path-lite optimization, the same `tile_00` Part4 benchmark on this device was about:

- `P4_BENCH iter=3 total_ms=3740` for one tile

After the optimization, the live room-creation run shows steady tiles at about:

- `1455-1472ms` per tile

That is roughly:

- about `2.5x` faster for the hot `Part4b` tile path
- about `60%` lower Part4b tile latency

Because Part1, Part3, and Part4a are still large, the whole room is not `2.5x` faster end-to-end. In practice, this working flow brings full room creation down to about `88s` on the reference device, which is roughly a `30%` end-to-end improvement versus the earlier pre-optimization Part4 estimate.

## Output quality note

This is the current output-oriented reference path.

- The run completed successfully.
- The user manually validated the visual result as strong.
- No obvious tiled Part4b artifacting was reported with this settings combination.

## If this flow breaks

Use:

```bash
adb logcat -v threadtime -s 'sharp_executorch_full:*' 'executorch:*' 'AndroidRuntime:*' 'libc:*'
```

Look for:

- routing falling away from `fine split tile_00`
- missing `loaded fine split tile_00 modules`
- shape-mismatch logs from `sharp_executorch_full`
- native aborts under `AndroidRuntime` or `libc`

For in-app diagnostics, use:

- `Vulkan & ExecuTorch diagnostics -> Run`
- `Part4 tile_00 fine split test -> Inspect`
- `Part4 tile_00 fine split test -> Benchmark 3x`

The `Inspect` screen should show the active `model_variant` plus delegate/kernel diagnostics from the `.manifest.json` sidecars.
