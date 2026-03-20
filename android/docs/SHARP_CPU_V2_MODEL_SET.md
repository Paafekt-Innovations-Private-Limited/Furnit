# Canonical CPU split set (LaCie `march10th2026/v2`)

This is the **six required files** for export v2, plus an **optional seventh** single Part4b for better visuals.

| Role | Basename |
|------|-----------|
| Part1+2 encoder (INT8) | `sharp_split_part1_int8.pte`, `sharp_split_part2_int8.pte` |
| Part3 image encoder (INT8) | `sharp_split_part3_int8.pte` |
| Part4a ViT chunks | `sharp_split_part4a_chunk_512.pte`, `sharp_split_part4a_chunk_65.pte` |
| Part4b Gaussian decoder (tiled, required for v2-minimal) | **`sharp_split_part4b_tile_b4.pte`** (4× batched tiles, 4 forwards) |
| Part4b single (**optional**, recommended) | `sharp_split_part4b_int8.pte` or `sharp_split_part4b_fp16.pte` or `sharp_split_part4b.pte` |

## Why names look “pattern-based”

ExecuTorch loads graphs by **fixed path** (`Module(path)`). The runtime does not guess roles from directory scan — every stage is a distinct exported graph, so the app and C++ use **stable basenames** agreed with `export_sharp_executorch_split4.py` (and friends).

Kotlin **`SharpExecuTorchSplitModelNames`** is the single place for `findFile(...)` strings on CPU. C++ uses the same literals in `sharp_executorch_full*.cpp` / `sharp_executorch_full_common.cpp`; rename an export only after updating **both**.

## Routing: Stable Part4b (single) + optional `.pte`

- **Stable ON + single Part4b on device:** Kotlin sets `preferSinglePart4b=true`; C++ **skips** tiled paths and runs the single decoder (INT8 → FP16 → FP32 by filename). You can keep `tile_b4` on disk unused.
- **Stable ON but no single file:** `preferSinglePart4b=false`; C++ uses **batched tiled** `tile_b4` first.

Deploy script `./deploy_sharp_v2_to_models_cpu.sh` pushes the six required files and **any** of the three optional single Part4b names found in the source folder.

## Example timings (CPU, tiled `tile_b4` only — your Mar 2026 log)

Rough breakdown from one phone run (no single Part4b, all XNNPACK):

| Stage | Time |
|-------|------|
| Part1+2 (35 patches) | **~74 s** |
| Part3 | **~1.6 s** |
| Part4a (512 + 65) | **~20 s** total |
| Part4b tiled (4× batch) | **~112 s** |
| Prune to 500k Gaussians | **~1.8 s** |
| **C++ full pipeline** (`[C++ FULL] … ms`) | **~210 s (~3.5 min)** |

Single Part4b replaces the **~112 s** tiled block with one forward (duration depends on model size/device; often less fog than 16-tile INT8).

## Visual quality (foggy 4×4 grid)

INT8 **tiled** Part4b (`tile_b4`) can look like **hazy squares** aligned with the 384×384 tile grid — quantization + per-tile decoding, not necessarily a bug in stitching. Mitigations:

1. **Add a single Part4b** (`sharp_split_part4b_int8.pte` or FP32/FP16 `sharp_split_part4b*.pte`) beside the v2 files; leave **Stable Part4b (single)** ON so the native path prefers the single decoder (higher RAM, usually clearer).
2. **ExecuTorch Vulkan** + Vulkan Part3/4 `.pte` in `models_vulkan` for faster, often cleaner Part4.
3. If patches look **shifted / sheared** (not foggy), try Settings **Swap tile NDC X/Y** — transposed exports need `swapTileNdcXY` passed into `runFullPipelineInt8Native`.

See also `docs/TEST_INT8_IN_APP.md` (tiled vs single).

## If you still see `runPart4bBatchedTiledPipeline` / `Part4b batch 1/4`

The v2 folder is **six files** and has **no** single Part4b unless you add it. Native logs **`SHARP_pte_paths`** with `Part4b_single=(none yet / tiled path)` and logcat **`Stable Part4b is ON but no single decoder`** until you push **`sharp_split_part4b.pte`** (chunked export) or **`sharp_split_part4b_int8.pte`** / **`sharp_split_part4b_fp16.pte`** into the same `models_cpu` directory as `tile_b4`.

Verify on device (debuggable):

`adb shell run-as com.furnit.android ls -la files/models_cpu | grep part4b`

## Sync

**Local folder (same name as on device):** `android/models_cpu/` — copy exports here, then deploy from that path. See **`models_cpu/README.md`**.

```bash
cd android
./copy_from_lacie_and_push_cpu_models.sh              # → models_cpu/ + adb push
# or
./fresh_sync_cpu_models_from_lacie.sh                   # default: LaCie v2
./fresh_sync_cpu_models_from_lacie.sh "$(pwd)/models_cpu"
# Optional: place sharp_split_part4b.pte (or _int8 / _fp16) in the source folder for Stable single Part4b.
```
