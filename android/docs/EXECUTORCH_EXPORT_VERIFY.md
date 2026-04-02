# ExecuTorch export vs app: prove export is broken first

If the pipeline **fails in app** but the same conceptual pipeline is fine outside the app, the **export/runtime boundary** is the first thing to blame.

## What usually goes wrong

1. **Graph exported with ops the app runtime/backend cannot execute** — export succeeds, app loads `.pte`, run fails on unsupported/lowered op.
2. **Mismatch between export-time assumptions and runtime input** — shape fixed differently, dtype mismatch (fp16 vs fp32), dynamic dim frozen wrong.
3. **Bad partitioning** — Part1/Part2 boundary tensor not what app expects; delegate got a broken subgraph.
4. **Quantization/export metadata** — scales/zero-points not handled as expected (portable vs backend).
5. **`.pte` is bad for this runtime** — load works, invoke dies.

## Best test: same `.pte`, same input, 4 checkpoints

- Python eager output  
- Exported `.pte` **portable** output  
- Exported `.pte` **backend/delegate** output (e.g. Vulkan)  
- App output / failure point  

If eager works but `.pte` fails before app-specific logic, **export is the issue**.

## Fastest way to corner it

For each part (Part1, Part2):

- Save one **known input** tensor to disk.
- Save **eager output** to disk.
- Run the **exported artifact** on that exact tensor.
- Compare: shape, dtype, min/max, first 8 values.  
- Log only: input shape/dtype, output shape/dtype, first 8 values, min/max, **error code**.

## Python verification and Vulkan backend

**To run Vulkan-exported `.pte` in Python**, the ExecuTorch **runtime** must have the **VulkanBackend** registered. The default `pip install executorch` package often does **not** include Vulkan; the Vulkan delegate is usually built for Android (NDK) or from source with `EXECUTORCH_BUILD_VULKAN=ON`.

- **If you need to verify Vulkan .pte in Python:** Build ExecuTorch from source with Vulkan enabled for the host (see ExecuTorch docs) and use that environment so the Python runtime links/registers VulkanBackend.
- **Otherwise:** Use **portable** .pte for Python verification: export Part1/Part2 with `--backend portable` (e.g. `export_part12_portable.sh` or `--part12-only-portable`), then run `verify_export_part1_part2.py --portable-only`. That proves **graph correctness** (eager vs portable .pte). Vulkan-specific issues (device/driver, shaders) remain app/device-side.

The verify script prints a hint if execution fails with a backend-not-registered style error when using a Vulkan-named .pte.

**Part-1-only debugging (cleanest first step):** Export and test **only Part 1** so Part 2/3/4 are not involved.

- **Export Part 1 only (portable FP32, no Vulkan):**
  ```bash
  python export_sharp_executorch_split4.py --part1-only --backend portable --dtype fp32 --output-dir executorch_models
  ```
- **Export Part 1 only (Vulkan FP16):**
  ```bash
  python export_sharp_executorch_split4.py --part1-only --backend vulkan --dtype fp16 --output-dir executorch_models
  ```
- With `--part1-only`, the export script also writes:
  - `part1_test_patch.pt`, `part1_test_patch_f32.bin` (and `part1_test_patch_f16.bin` if fp16)
  - `part1_tokens_golden_f32.bin`, `part1_block5_golden_f32.bin` (eager Part1 on that patch)
  - Prints shape, dtype, min/max/mean, first 8 values for tokens and block5.
- **In the app:** Feed the same patch (e.g. load from `part1_test_patch_f32.bin` or copy from .pt), run Part 1 .pte, then log: input shape/dtype, output0/output1 shapes, first 8 values of each, min/max/mean. Compare to the printed goldens. If shapes or values differ, the issue is export/runtime or layout/dtype; if they match, move on to Part 2.
- **Fixtures without re-export:** Run `python generate_part1_test_fixtures.py --output-dir executorch_models` (optionally `--fp16`) to regenerate only the test patch and golden .bin files.

**On-device verification:** You can skip Python .pte execution and verify directly on the Android device: export on host, push the `.pte` files with `push_sharp_executorch_models.sh` (or `push_sharp_cpuvulkan_hybrid_androidstudio.sh`), then run the app. If the app loads and runs Part1/Part2 without BackendFailed or crash, the export/runtime match on device. Use logcat (e.g. `sharp_executorch_full`, `VulkanDiag`) to confirm forward completion and any error codes.

**Location:** `android/verify_export_part1_part2.py`

**What it does:**

- **Test A:** Part1 eager vs Part1 `.pte` on the same patch input `[B, 3, 384, 384]`.
- **Test B:** Part2 eager vs Part2 `.pte` on the same tokens input (Part1 eager output) `[B, 577, 1024]`.
- Saves fixtures to `android/verify_export_fixtures/` (part1_input.pt, part1_eager_tokens.pt, part1_eager_block5.pt, part2_input.pt, part2_eager_output.pt).
- Logs: input/output shape and dtype, first 8 values, min/max, and any .pte execution error.

**Usage:**

```bash
cd android
# Full run (Part1 + Part2); needs SHARP weights and third_party/ml-sharp
python verify_export_part1_part2.py --output-dir executorch_models

# Only Part1
python verify_export_part1_part2.py --part1-only

# Only Part2 (uses saved Part1 output from fixtures)
python verify_export_part1_part2.py --part2-only

# Batch=2 (Vulkan app path: sharp_split_part1_b2_vulkan_fp16.pte)
python verify_export_part1_part2.py --batch 2

# Portable only (no Vulkan backend required in Python; use after exporting Part1/2 with --backend portable)
python verify_export_part1_part2.py --portable-only
```

**Interpretation:**

- **Part1 .pte error** → Part1 export or delegate is broken.
- **Part2 .pte error** with known-good Part1 output → Part2 export or delegate is broken.
- **Both pass in Python but app fails** → boundary packing / tensor handoff / app input formatting.

## Exact export command (Vulkan)

```bash
cd android
python export_sharp_executorch_split4.py \
  --backend vulkan \
  --chunked-part4 \
  --dtype fp16 \
  --patch-batch-size 2 \
  --sharp-src third_party/ml-sharp/src \
  --weights sharp_litert_models/sharp_2572gikvuh.pt \
  --output-dir executorch_models
```

Or use the wrapper (writes full log to `export_log_vulkan_*.txt`):

```bash
./export_sharp_executorch_with_log.sh
```

## App input tensor shapes (C++ pipeline)

| Stage   | Input shape        | Output / note |
|--------|--------------------|----------------|
| Part1  | `[B, 3, 384, 384]` | B=1 or B=2 (Vulkan: b2). Output: tokens `[B, 577, 1024]`, block5 `[B, 577, 1024]`. |
| Part2  | `[B, 577, 1024]`   | Output: `[B, 1024, 24, 24]`. |

App uses **batch 2** for Vulkan when files `sharp_split_part1_b2_vulkan_fp16.pte` / `sharp_split_part2_b2_vulkan_fp16.pte` exist; otherwise batch 1 with `sharp_split_part1_vulkan_fp16.pte` / `sharp_split_part2_vulkan_fp16.pte`.

## Portable vs backend

- **Portable `.pte` runs, Vulkan `.pte` fails** → Vulkan delegate or export lowering is wrong (ops/driver).
- **Both fail** → export (graph/shape/dtype) or runtime load is wrong.
- **Both pass in Python, app fails** → app input prep, tensor handoff, or app-side backend config.

## Extra sign export is broken

If the app dies at **invocation** with no Java-side exception and no obvious memory crash, that usually means:

- Runtime hit invalid lowered graph.
- Backend rejected node/tensor layout.
- Exported artifact carries something the runtime cannot honor.

Then debug the **`.pte` artifact** (e.g. with `verify_export_part1_part2.py` and optional portable export) before changing app code.
