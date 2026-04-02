# `models_cpuvulkan_hybrid` — hybrid model dir (CPU Part1+2 + Vulkan Part3–4)

## Always required (room / `inferStreaming`)

| Files | Role |
|-------|------|
| `sharp_split_part1_int8.pte` | Part1 on CPU (hybrid sidecar) |
| `sharp_split_part2_int8.pte` | Part2 on CPU (hybrid sidecar) |
| `sharp_split_part3_vulkan_fp32.pte` or `_vulkan_fp16.pte` | Part3 on Vulkan |
| `sharp_split_part4a_chunk_512_vulkan.pte` | Part4a 512 |
| `sharp_split_part4a_chunk_65_vulkan.pte` | Part4a 65 |

## Part4b — pick **one complete** strategy (tiled-only in Vulkan C++)

Native order: **`runPart4bBatchedTiledPipeline`** first, then **`runPart4bTiledFullPipeline`** (`sharp_executorch_full_vulkan.cpp` → `sharp_executorch_full_common.cpp`).

If the **fine-split tile_00** five-pack exists, the **batched** path **skips** itself so the **sequential** path can run fine-split first.

Kotlin `part4bSatisfied` must match the same sets (including fine-split — see `ExecutorchInt8Sharp.hasPart4bNonSingleDecoder`).

### A. Fine-split **tile_00** (5 files) — preferred in sequential path

- `sharp_split_part4b_tile_00_stage_pre_vulkan.pte`
- `sharp_split_part4b_tile_00_decoder_head.pte`
- `sharp_split_part4b_tile_00_init_base.pte`
- `sharp_split_part4b_tile_00_raw_heads_vulkan.pte`
- `sharp_split_part4b_tile_00_compose.pte`

### B. Split **tile_00** (4 files)

- `sharp_split_part4b_tile_00_stage_a_vulkan.pte`
- `sharp_split_part4b_tile_00_init_base.pte`
- `sharp_split_part4b_tile_00_raw_heads_vulkan.pte`
- `sharp_split_part4b_tile_00_compose.pte`

### C. Fine-split **tile_b2** (5 files) — batched path

- `sharp_split_part4b_tile_b2_stage_pre_vulkan.pte`
- `sharp_split_part4b_tile_b2_decoder_head.pte`
- `sharp_split_part4b_tile_b2_init_base.pte`
- `sharp_split_part4b_tile_b2_raw_heads_vulkan.pte`
- `sharp_split_part4b_tile_b2_compose.pte`

### D. Split **tile_b2** (4 files)

- `sharp_split_part4b_tile_b2_stage_a_vulkan.pte`
- `sharp_split_part4b_tile_b2_init_base.pte`
- `sharp_split_part4b_tile_b2_raw_heads_vulkan.pte`
- `sharp_split_part4b_tile_b2_compose.pte`

### E. Legacy **batched** one-file (simplest count)

- `sharp_split_part4b_tile_b2.pte` **or** `sharp_split_part4b_tile_b4.pte`

### F. Legacy **sequential** one-file

- `sharp_split_part4b_tile_00.pte` **or** `sharp_split_part4b_tile_full.pte`

### Not used

- `sharp_split_part4b_vulkan.pte` — not loaded by Vulkan full pipeline.

---

## This folder’s **current** layout (repo)

Populated from **`android/sharp_vulkan_only/`** (Vulkan + **strategy A** fine-split tile_00) and **`models_cpu` INT8** (e.g. LaCie backup). Re-run:

`./populate_models_cpuvulkan_hybrid_from_backups.sh`

(`VK_SRC` defaults to `android/sharp_vulkan_only`; set `CPU_SRC` if LaCie is elsewhere.)

Other Part4b strategies (B–F) are still valid — swap in that file set and remove unused Part4b `.pte` from this folder to avoid confusion.

Push for Android Studio installs (default source is `models_cpuvulkan_hybrid/`; skips large `sharp_split_part1/2_vulkan_fp{16,32}.pte` not used in hybrid): `./push_sharp_cpuvulkan_hybrid_androidstudio.sh`

`.pte` files are gitignored.
