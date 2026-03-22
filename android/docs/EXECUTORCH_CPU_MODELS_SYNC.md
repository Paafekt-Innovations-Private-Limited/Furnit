# ExecuTorch CPU split models — sync without Part4b mismatch

The C++ full pipeline (`sharp_executorch_full`) loads **Part3 → Part4a → Part4b** from the same directory. If **Part4b** (`.pte`) comes from a **different export** than Parts 1–3 / 4a, `forward()` on Part4b often returns **error 18** = **`InvalidArgument` (0x12)** — wrong tensor shapes vs graph.

## Where files live on device

- **etCpu** flavor: `Android/data/com.furnit.android/files/models_cpu/` (external) and app **internal** `files/models_cpu/` after sync.
- The app copies `sharp_split*.pte` from APK assets (when bundled) and from scoped external → internal (`hydrateBundledAndExternalModels` / `initialize`).

### No legacy `files/models` for `sharp_split*.pte`

**`sharp_split*.pte` are only read under `models_cpu` / `models_cpuvulkan_hybrid`** (internal then external). Stale copies under `files/models/` are ignored. Push the working set to **`models_cpu`** or **`models_cpuvulkan_hybrid`**.

## Required workflow (recommended)

1. **Clear** old `.pte` so you never mix a new encoder with an old decoder:

   ```bash
   cd android
   ./clear_device_models_cpu.sh
   ```

   If `run-as` fails (release build), use **Settings → Apps → Furnit → Clear storage** once.

2. **Push one complete export** (same `export_sharp_executorch_split4.py` run / same folder):

   ```bash
   ./push_sharp_executorch_cpu_models.sh /path/to/your/executorch_models
   ```

3. **From LaCie** — default export folder in scripts: **`/Volumes/LaCie/march10th2026/v2`** (override with first arg).

   ```bash
   ./fresh_sync_cpu_models_from_lacie.sh
   # or another folder:
   ./copy_from_lacie_and_push_cpu_models.sh /Volumes/LaCie/other_export
   ```

   `copy_from_lacie_and_push_cpu_models.sh` **requires** on the source dir:

   - `sharp_split_part1/2/3_int8.pte`, `sharp_split_part4a_chunk_512.pte`, `sharp_split_part4a_chunk_65.pte`
   - **and** at least one of `sharp_split_part4b_int8.pte` or `sharp_split_part4b.pte`

   **Optional** (copied if present): `part1/2_b4_int8.pte`, single `part4b` / `part4b_int8`, other tiles, portable fp16 fallbacks.

   **v2 six-file set** (single Part4b not required): see **`docs/SHARP_CPU_V2_MODEL_SET.md`** — uses **`sharp_split_part4b_tile_b4.pte`** only; Kotlin + C++ must treat `tile_b4` as a valid Part4b path (not only `tile_00` / `tile_full`).

## Log reference

| Log | Meaning |
|-----|--------|
| `InvalidArgument(0x12)` | Inputs to Part4b don’t match this `.pte` — usually **mixed export set**. |
| `Part4b forward fail … sharp_split_part4b.pte` | FP32 Part4b path; prefer **`sharp_split_part4b_int8.pte`** from the **same** export as `part3_int8`. |

## Vulkan note

Warnings that *mention* Vulkan during a **CPU** run are only **performance hints** for slow Part4a/512; they are not Vulkan error codes.
