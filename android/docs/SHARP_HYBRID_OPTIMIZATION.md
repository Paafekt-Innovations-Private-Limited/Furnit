# SHARP hybrid CPU→Vulkan handoff (experimental)

Runtime flags (SharedPreferences `furnit_prefs`, **default false**):

| Key | Effect |
|-----|--------|
| `sharp_hybrid_overlap_part3` | Start Part3 (full-image ViT) on a **worker thread** while Part1+2 runs; join after Part1+2. Hides Part3 load/forward behind encoder when it works; may increase peak GPU contention—validate per device. |
| `sharp_hybrid_preload_part4b_tile00` | **Async mmap+load** of tiled Part4b `tile_00` split modules during Part4a (512+65); consumed on first `runPart4bTiledFullPipeline` if `modelDir` matches. |
| `sharp_hybrid_interleave_part12` | Vulkan **25-only** path: run Part1→Part2 **per patch** without releasing Part1 between patches (lower latency vs two-pass; higher peak memory). Only applied if `MemAvailable` ≥ **512 MB** (see Kotlin `HYBRID_INTERLEAVE_MIN_AVAIL_BYTES`). |

All paths remain **opt-in**; defaults preserve the previous two-pass encoder + sequential Part3 behavior.

## Native logs (grep)

- `[HYBRID]` — which optimizations activated; Part3 overlap / Part4b preload milestones.
- `[HYBRID_TIMING]` — one line per run: `part12_ms`, `part3_ms`, `part4a_ms`, `total_ms`, flags, `availKb`.
- `[TIMING] Part1+2 wall` — encoder wall time and whether two-pass or interleaved was used.

## Benchmarking (validate-perf-memory)

1. **Baseline:** clear hybrid prefs (all false). Capture logcat for one room: note `[HYBRID_TIMING]` and total room-creation time.
2. **Single flag:** enable one experimental flag at a time; compare `part12_ms` / `part3_ms` / `part4a_ms` / total.
3. **Memory:** on a **low-RAM** device, keep interleave off; confirm no OOM with overlap Part3 / preload Part4b.
4. **High-RAM:** try interleave + overlap; watch for `VK_ERROR_DEVICE_LOST` or GPU TDR—back off flags.

## Guardrails (unchanged)

- Vulkan batch-2 Part1+2 stays disabled (ExecuTorch `Tensor.cpp` boundary issue).
- Part4b stays **tiled / split** only in Vulkan; no monolithic unsafe Vulkan Part4b.
