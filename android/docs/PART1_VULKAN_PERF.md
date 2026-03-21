# Part1 Vulkan performance (Furnit)

Focus **only Part1** (`sharp_split_part1.pte`) until forward time is acceptable. Registration and “keep Module alive” are necessary but **not sufficient** if each `forward()` still costs many minutes: that usually means **per-inference** cost (layout churn, attention on Vulkan, repacking), not a missing warmup.

## March 21, 2026 runtime cleanup: problem solved

The technical problem fixed in app native code was **dead encoder prep work in the current fixed 25-only path**, plus **excess hot-loop logging**.

Problem:

- The Android full pipeline currently forces `part12_25_only=true` for room creation.
- In that mode, the app does **not** run the `0.5x` or `0.25x` Part1+2 patch passes.
- Even so, native code was still downsampling the full input image to `0.5x` and `0.25x` buffers before Part1+2 started.
- The steady-state Part1+2 loop also emitted extra per-patch debug logs around readback and Part2 calls.

Why that was bad:

- The image downsample work consumed CPU time and memory bandwidth on every run, but its outputs were never used in `25-only` mode.
- The extra logging added avoidable overhead in the hottest encoder loop.
- This was runtime overhead, not model math.

What changed:

- `sharp_executorch_full_vulkan.cpp` now skips `0.5x` / `0.25x` image downsample prep when `part12_25_only` is active.
- `sharp_executorch_full.cpp` does the same for CPU / hybrid Part1+2 runs.
- The hottest per-patch Part1+2 debug logs were removed, while first-patch crash-triage logs and all error logs were kept.

What this did **not** solve:

- It did **not** change ExecuTorch itself.
- It did **not** change exported `.pte` files or Part1/Part2 graph partitioning.
- It did **not** remove the real remaining encoder bottleneck: Part1/Part2 forward cost is still dominated by model/backend work.

Current verified result from a real run after this cleanup:

- `part12OnCpu=1` in the log, so this was the current **hybrid** flow: Part1+2 on CPU, Part3/4 on Vulkan.
- `runFullPipelineInt8` start: `10:17:53.658`
- `JNI RETURN` validated: `10:19:20.625`
- Native pipeline total: `86.967s`
- Part1+2 patches: `30.478s`

So the cleanup removed waste, but Part1+2 is still a major stage and still needs deeper model/export-side work for a large next gain.

## Concrete next profiling plan

Use this order:

1. Check the room log first.
   If it says `part12OnCpu=1`, that run is hybrid and does **not** tell you about Vulkan Part1+2 speed.

2. Benchmark standalone Part1 in-app.
   Use `Warmup` then `Benchmark 3×`, and capture `P1_BENCH`, `PART1_RUN`, and `PART1_ARTIFACT`.

3. Compare Vulkan Part1 vs portable Part1 on the same patch.
   This tells you whether Vulkan is helping at all before you spend time on room-level tuning.

4. If standalone Vulkan Part1 is still bad, collect ETDump for standalone Part1 and Part2 with `executor_runner`.
   Read the Inspector output as:
   - big math ops dominate -> split / repartition around attention-heavy blocks
   - many `view` / `permute` / `concat` / `texture3d` ops dominate -> layout/storage churn is the problem
   - many graph breaks / fallback -> delegated region is too ambitious

See `docs/EXECUTORCH_VULKAN_PROFILING.md` for the concrete ETDump workflow and interpretation rules.

## On-device: three timed forwards (same PID)

1. Install/run the app; **do not force-stop** (same process).
2. **Settings → Developer → Benchmark 3×** (or Warmup first if you want load+2× warmup done separately).
3. Capture logs:

```bash
adb logcat -d | grep P1_BENCH
adb logcat -d | grep PART1_RUN
adb logcat -d | grep PART1_GOLDEN
adb logcat -d | grep PART1_ARTIFACT
```

Lines include:

- `P1_BENCH session_ensure_ms=…` — includes **load + 2× warmup** only when the session was cold.
- `P1_BENCH timed_forward 1/3 duration_ms=…` (and 2/3, 3/3).
- `P1_BENCH summary duration_ms=[a,b,c] ratio_2_over_1=… ratio_3_over_1=…`

**Read results:**

- **Run 1 huge, runs 2–3 much smaller** → first forward pays pipeline/shader; later work reuses caches.
- **All three similar and huge** → steady-state execution is slow (graph/partition/layout), not “missing second warmup.”

Compare with **portable Part1** (`--part12-only-portable` or `--part1-only` + `portable`) for a fast CPU baseline on the same patch.

## Export-side levers (Part1 only)

Script: `android/export_sharp_executorch_split4.py` (`--part1-only`, `--backend vulkan|portable`).

1. **Fewer Vulkan layout transitions**  
   Re-export and compare exporter output: fewer “Inserting transition(s)” / buffer↔texture repacks is better. Large delegated graphs with many `view`/`permute`/`linear`/`bmm`/`softmax` boundaries often bounce **WIDTH_PACKED** vs **CHANNELS_PACKED**.

2. **Partition size tuning**  
   Try **smaller** Vulkan regions (more CPU fallback) vs **maximal** Vulkan — “more Vulkan” is not always faster if transitions dominate. Compare `[Partition] Vulkan strings in .pte` and real device times.

3. **Split the model (future export work)**  
   Part1a (early blocks) vs Part1b (attention-heavy) as separate `.pte` files lets you see which half burns time on Vulkan.

4. **One canonical artifact**  
   For tuning, stick to one Part1 file name/path (`sharp_split_part1.pte`) so the app cache and driver behavior stay comparable.

5. **Release + ExecuTorch version**  
   Build native/ExecuTorch **Release** for timing. Vulkan backend evolves quickly — newer ExecuTorch + re-export may change transition counts and perf.

## App behavior (already in Furnit)

- **Persistent `Module`** for Part1 test path; **double warmup** after load; **Run** reuses cache.
- **Release Part1 cache** in Settings if you swap `.pte` on disk.

See also: `android/docs/EXECUTORCH_VULKAN_REGISTRATION.md` (registration / load order).
