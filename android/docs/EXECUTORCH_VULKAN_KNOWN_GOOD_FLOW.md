# Known-Good ExecuTorch Vulkan Room Creation Flow

This document records the current working Vulkan room-creation path for SHARP in Furnit.

Date of the reference run: March 21, 2026.

This is the flow that currently gives:

- stable room creation on the `etVulkan` APK
- the optimized `tile_00` Part4 path
- visually strong room output on a real indoor capture

## Build and model set

- APK flavor: `etVulkan`
- Models directory on device: `files/models_vulkan`
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

## Settings to use

Open `Profile -> Settings -> Developer` and use:

| Setting | Value |
|---|---|
| `Inference Backend` | `ExecuTorch INT8 (Vulkan)` |
| `Max Gaussians` | `All` |
| `Use true 1280x1280` | `OFF` |
| `Prefer Vulkan FP16 models` | `ON` |
| `Prefer single Part4b` | `OFF` |
| `Record ETDump on next room creation` | `OFF` for normal runs |

Notes:

- `Prefer single Part4b = OFF` is intentional. The current known-good Vulkan path is the fine-split `tile_00` route, not the old single-decoder route.
- `Use true 1280x1280 = OFF` keeps the currently validated `1536` hybrid split path.

## Working flow

1. Build and install the `etVulkan` APK.
2. Push the Vulkan SHARP model pack to `models_vulkan`.
3. Set the toggles above in Settings.
4. Create a room from a single photo as usual in the app.
5. Let the pipeline complete without changing backend toggles mid-run.

## What the logs should show

The important routing lines are:

```text
runFullPipelineInt8: modelDir=.../models_vulkan useVulkan=1 ... preferSingleP4b=0
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
